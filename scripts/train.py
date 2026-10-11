"""
Train the Task 2 model from features/ (plan.md §4.1–4.7): one GBT that takes the threshold as an
input feature and estimates P(rain in the next 30 min > threshold) for a station at time t.

Readings come in 0.2 mm steps, so "> 3.1 mm" means exactly "> 3.0 mm": every threshold rounds down to the grid
0, 0.2, ..., 30 mm (index j = threshold / 0.2) without changing the answer. Each training row is copied COPIES
times with random grid thresholds (small amounts more often), so the model learns every step. Validation,
2023 and 2024 rows are copied once per EVAL_THRESHOLDS, where metrics are reported.

  --mode tune    Rolling-origin folds inside 2017–2022 (validate on 2020, 2021, 2022). Tries every combination of
                 the --max-depth, --max-iter, --step-size, --max-bins and --min-instances values given, and
                 reports Brier and AUCs per fold for each GBT setting and both baselines, ranked by mean Brier.
                 A setting is better only if it beats another by more than the fold-to-fold spread.
                 Run once per ablation (--drop).
                 Writes <bucket>/tune/<run>/.
  --mode final   Fit on 2017–2022 and calibrate (isotonic) on 2023. Writes <bucket>/models/ (the GBT, the
                 calibrator and the climatology counts). 2024 is not touched: evaluate it with
                 `predict.py --evaluate`, which picks the CSI cut-offs on 2023 and writes <bucket>/results/.

--sample-days keeps a share of whole training days for fitting the GBT (both modes). Climatology always uses
every training day; validation, 2023 and 2024 are never sampled.

Run:
  scripts/run_script.sh train -- --mode tune --sample-days 0.2 --max-depth 4 5 7 --max-iter 100 200
  scripts/run_script.sh train -- --mode tune --sample-days 0.2 --drop day_of_year
  scripts/run_script.sh train -- --mode final --sample-days 0.5 --max-depth 6
"""

import argparse
import itertools
import math
import os
import statistics
from functools import reduce

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import SparkSession, Window

from build_features import FEATURE_GROUPS
from utils import BUCKET_NAME

GRID_STEPS = 150                        # thresholds 0, 0.2, ..., 30 mm; anything above answers as "> 30 mm"
COPIES = 3                              # random thresholds per training row
EVAL_THRESHOLDS = [0.0, 0.2, 1.0, 5.0, 10.0]    # mm; 0 means "any rain"
CLIM_PRIOR_WEIGHT = 100                 # pseudo-readings pulling a climatology cell toward the overall rate
TRAIN_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
FOLD_VAL_YEARS = [2020, 2021, 2022]     # fold k trains on 2017 .. val_year - 1
CALIB_YEAR, TEST_YEAR = 2023, 2024
DAY_S = 24 * 3600
MODELS = ("clim", "pers", "gbt")        # climatology and persistence baselines, then the GBT
# GBT settings: (name in results, command-line dest, Spark param). Tune mode tries every combination.
GBT_PARAMS = [("depth", "max_depth", "maxDepth"), ("iter", "max_iter", "maxIter"),
              ("step", "step_size", "stepSize"), ("bins", "max_bins", "maxBins"),
              ("min_leaf", "min_instances", "minInstancesPerNode")]


def grid_index(mm):
    """Grid index of a threshold in mm, rounded down (exact: readings are 0.2 mm steps) and capped."""
    return min(int(math.floor(mm * 5 + 1e-9)), GRID_STEPS)


def feature_columns(drop=()):
    """Model inputs, leaving out the groups in `drop` (for ablation). The threshold is always an input."""
    return [c for g, cs in FEATURE_GROUPS.items() if g not in drop for c in cs] + ["threshold"]


def random_thresholds():
    """COPIES grid indices per row, log-uniform so small amounts are common but every step appears:
    P(j) is proportional to log((j + 2) / (j + 1)). Seeded by the row, so reruns draw the same."""
    def draw(c):
        u = F.pmod(F.xxhash64("station_id", "bucket", F.lit(c)), F.lit(1_000_000)) / 1_000_000
        return F.floor(F.exp(u * math.log(GRID_STEPS + 2))).cast("int") - 1
    return F.array_distinct(F.array(*[draw(c) for c in range(COPIES)]))


def fixed_thresholds():
    return F.array(*[F.lit(grid_index(t)) for t in EVAL_THRESHOLDS])


def expand(rows, js):
    """One row per grid index in array column `js`: `threshold` (a feature), `label` = rain_next_30m > threshold
    and `pers` (persistence: the last 30 min exceeded it). Compared as integer grid steps, so exact."""
    return (rows.withColumn("j", F.explode(js))
            .withColumns({"threshold": F.col("j") / 5,
                          "label": (F.round(F.col("rain_next_30m") * 5) > F.col("j")).cast("int"),
                          "pers": (F.round(F.col("rain_30m") * 5) > F.col("j")).cast("double")}))


def climatology_counts(rows):
    """Per (station_id, year, month, hour): n readings and k, where k[j] counts the readings with more than
    j * 0.2 mm in the next 30 min, for every grid step j."""
    keys = ["station_id", "year", "month", "hour"]
    hist = (rows.groupBy(*keys, F.round(F.col("rain_next_30m") * 5).cast("int").alias("i"))
            .agg(F.count(F.lit(1)).alias("c")))
    k = F.transform(F.sequence(F.lit(0), F.lit(GRID_STEPS)),
                    lambda j: F.aggregate(F.filter("h", lambda x: x["i"] > j), F.lit(0).cast("long"),
                                          lambda acc, x: acc + x["c"]))
    return (hist.groupBy(*keys)
            .agg(F.sum("c").alias("n"), F.collect_list(F.struct("i", "c")).alias("h"))
            .select(*keys, "n", k.alias("k")))


def add_arrays(arrays):
    """Element-wise sum of a collected list of k arrays."""
    zero = F.array_repeat(F.lit(0).cast("long"), GRID_STEPS + 1)
    return F.aggregate(arrays, zero, lambda acc, a: F.zip_with(acc, a, lambda x, y: x + y))


def climatology(rows, counts, train_years, weight=CLIM_PRIOR_WEIGHT):
    """Add clim: the rate of exceeding the row's threshold (column j) for its station, month and hour, from
    `train_years` only (plan.md §4.4). Rows in a training year get the rate from the OTHER training years
    (leave-one-year-out); every other row gets the rate from all of them. Each cell is smoothed toward the
    overall training rate with `weight` pseudo-readings, so sparse cells (new stations) stay sane."""
    keys = ["station_id", "month", "hour"]
    train = counts.where(F.col("year").isin(train_years))
    overall = train.agg(F.sum("n").alias("n"), add_arrays(F.collect_list("k")).alias("k")).first()
    prior = F.array(*[F.lit(k / overall["n"]) for k in overall["k"]])
    total = train.groupBy(*keys).agg(F.sum("n").alias("cell_n"), add_arrays(F.collect_list("k")).alias("cell_k"))
    own = train.select(*keys, "year", F.col("n").alias("own_n"), F.col("k").alias("own_k"))

    # own_* is null outside the training years, so those rows keep the full count.
    i = F.col("j") + 1
    n = F.coalesce(F.col("cell_n"), F.lit(0)) - F.coalesce(F.col("own_n"), F.lit(0))
    k = (F.coalesce(F.element_at("cell_k", i), F.lit(0)) - F.coalesce(F.element_at("own_k", i), F.lit(0)))
    return (rows.join(F.broadcast(total), keys, "left").join(own, keys + ["year"], "left")
            .withColumn("clim", (k + weight * F.element_at(prior, i)) / (n + weight))
            .drop("cell_n", "cell_k", "own_n", "own_k"))


def trim(df, years):
    """Drop rows within one day of 1 January of each year in `years`, so no feature or label window
    crosses from one set into another (plan.md §4.3)."""
    t = F.unix_timestamp("bucket")
    near = [F.abs(t - F.unix_timestamp(F.lit(f"{y}-01-01 00:00:00"))) < DAY_S for y in years]
    return df.where(~reduce(lambda a, b: a | b, near))


def sample_days(df, frac):
    """Keep a fixed `frac` of whole days (never rows: neighbouring rows are near-copies, plan.md §4.3)."""
    return df if frac >= 1 else df.where(F.pmod(F.xxhash64(F.to_date("bucket")), F.lit(1000)) < frac * 1000)


def split(features, train_years, other_years, frac):
    """Training rows (day-sampled, random thresholds) and rows of each other year (EVAL_THRESHOLDS), each with
    climatology from the training years, trimmed at every boundary between them. Also returns the counts."""
    rows = trim(features.where(F.col("year").isin(train_years + other_years)), other_years)
    train = rows.where(F.col("year").isin(train_years))
    counts = climatology_counts(train).persist()
    return (climatology(expand(sample_days(train, frac), random_thresholds()), counts, train_years),
            [climatology(expand(rows.where(F.col("year") == y), fixed_thresholds()), counts, train_years)
             for y in other_years],
            counts)


def fit(train, drop, setting):
    """`setting` maps each GBT_PARAMS name to a value."""
    gbt = GBTClassifier(labelCol="label", featuresCol="x", seed=42,
                        **{param: setting[name] for name, _, param in GBT_PARAMS})
    return Pipeline(stages=[VectorAssembler(inputCols=feature_columns(drop), outputCol="x"), gbt]).fit(train)


def predict(model, df):
    """Add gbt: the GBT probability of exceeding the row's threshold."""
    return (model.transform(df)
            .withColumn("gbt", vector_to_array("probability")[1])
            .drop("x", "rawPrediction", "probability", "prediction"))


def summarise(df, p, y="label"):
    """Brier, ROC-AUC and PR-AUC of probability column `p` against label `y`, plus per-percent bins
    (count, positives, sum of p) for the reliability table and cut-off search."""
    rows = (df.groupBy(F.least(F.floor(F.col(p) * 100), F.lit(99)).cast("int").alias("b"))
            .agg(F.count(F.lit(1)).alias("n"), F.sum(y).alias("k"), F.sum(p).alias("s"),
                 F.sum((F.col(p) - F.col(y)) ** 2).alias("sq"))
            .collect())
    bins = {r.b: r for r in rows}
    ev = BinaryClassificationEvaluator(rawPredictionCol=p, labelCol=y)
    return {"brier": sum(r.sq for r in rows) / sum(r.n for r in rows),
            "roc_auc": ev.setMetricName("areaUnderROC").evaluate(df),
            "pr_auc": ev.setMetricName("areaUnderPR").evaluate(df)}, bins


def categorical(bins, cut):
    """POD, FAR, CSI and frequency bias when p >= cut / 100 means "yes"."""
    hits = sum(r.k for b, r in bins.items() if b >= cut)
    false_alarms = sum(r.n - r.k for b, r in bins.items() if b >= cut)
    misses = sum(r.k for r in bins.values()) - hits
    return {"cutoff": cut / 100,
            "pod": hits / max(hits + misses, 1),
            "far": false_alarms / max(hits + false_alarms, 1),
            "csi": hits / max(hits + misses + false_alarms, 1),
            "bias": (hits + false_alarms) / max(hits + misses, 1)}


def best_cut(bins):
    return max(range(1, 100), key=lambda c: categorical(bins, c)["csi"])


def ordering_violations(df, p):
    """Share of (station, t) where a higher threshold gets a higher probability than a lower one (plan.md §4.1).
    Trees don't force P to fall as the threshold rises, so this is measured, not assumed."""
    w = Window.partitionBy("station_id", "bucket").orderBy("j")
    up = F.coalesce(F.col(p) > F.lag(p).over(w), F.lit(False)).cast("int")
    return (df.withColumn("up", up).groupBy("station_id", "bucket").agg(F.max("up").alias("v"))
            .agg(F.avg("v")).first()[0])


def at(df, t):
    """Rows for one reported threshold."""
    return df.where(F.col("j") == grid_index(t))


def write_csv(spark, rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    df = spark.createDataFrame([[r.get(k) for k in keys] for r in rows], keys)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(path)
    print(f"wrote {path}")


def tune(spark, features, args, root):
    """Every combination of the GBT_PARAMS flags' values on every fold. Each fold's data is prepared once and
    reused across settings; the baselines don't depend on the setting, so they're scored once."""
    names = [name for name, _, _ in GBT_PARAMS]
    values = [getattr(args, a) for _, a, _ in GBT_PARAMS]
    settings = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    no_setting = dict.fromkeys(names)
    thresholds = ["all"] + [f"{t:g}" for t in EVAL_THRESHOLDS]   # "all" pools them: one number per setting
    results = []

    def score(val, fold, models, setting):
        broken = ordering_violations(val, "gbt") if "gbt" in models else None
        for t in thresholds:
            rows = val if t == "all" else at(val, float(t))
            for model in models:
                m, _ = summarise(rows, model)
                results.append({"fold": fold, **setting, "threshold": t, "model": model, **m,
                                "ordering_violations": broken if model == "gbt" else None})

    for v in FOLD_VAL_YEARS:
        train, (val,), counts = split(features, [y for y in TRAIN_YEARS if y < v], [v], args.sample_days)
        train = train.persist(StorageLevel.MEMORY_AND_DISK)
        val = val.persist(StorageLevel.MEMORY_AND_DISK)
        score(val, v, ("clim", "pers"), no_setting)
        for setting in settings:
            print(f"fold {v}: {setting}")
            scored = predict(fit(train, args.drop, setting), val).persist(StorageLevel.MEMORY_AND_DISK)
            score(scored, v, ("gbt",), setting)
            scored.unpersist()
        for df in (train, val, counts):
            df.unpersist()

    def mean_sd(model, setting, t):
        b = [r["brier"] for r in results if r["model"] == model and r["threshold"] == t
             and all(r[k] == setting[k] for k in names)]
        return statistics.mean(b), statistics.stdev(b)

    print("mean Brier ± fold spread (sd), by threshold (mm):", "  ".join(f"{t:>17}" for t in thresholds))
    rows = [(m, no_setting) for m in ("clim", "pers")] + [("gbt", s) for s in settings]
    for model, setting in sorted(rows, key=lambda r: mean_sd(*r, "all")[0]):
        name = model if model != "gbt" else "gbt " + " ".join(f"{k}={v:g}" for k, v in setting.items())
        print(f"  {name:<50}", "  ".join("%.5f ± %.5f" % mean_sd(model, setting, t) for t in thresholds))

    fmt = lambda xs: "-".join(f"{x:g}" for x in xs)
    run = "_".join(f"{name}{fmt(getattr(args, a))}" for name, a, _ in GBT_PARAMS)
    run += f"_days{args.sample_days:g}" + "".join(f"_no-{g}" for g in args.drop)
    write_csv(spark, results, f"{root}/tune/{run}")


def final(spark, features, args, root):
    """Fit the GBT on 2017–2022 and the calibrator on 2023, and save both with the climatology counts.
    Evaluation on 2024 is predict.py --evaluate, which only loads what this saves."""
    # TEST_YEAR is listed only so the 2023 rows are trimmed at the 2024 boundary too; its rows are never used here.
    train, (calib, _), counts = split(features, TRAIN_YEARS, [CALIB_YEAR, TEST_YEAR], args.sample_days)
    model = fit(train.persist(StorageLevel.MEMORY_AND_DISK), args.drop,
                {name: getattr(args, a)[0] for name, a, _ in GBT_PARAMS})
    model.write().overwrite().save(f"{root}/models/gbt")    # saved first: the fit is the expensive part
    # One calibrator for all thresholds: it is monotone, so it never breaks the ordering across thresholds.
    iso = IsotonicRegression(featuresCol="gbt", labelCol="label", predictionCol="cal").fit(predict(model, calib))
    iso.write().overwrite().save(f"{root}/models/isotonic")
    counts.write.mode("overwrite").parquet(f"{root}/models/clim_counts")
    print(f"wrote {root}/models/")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Task 2 GBT model.")
    parser.add_argument("--bucket", default=BUCKET_NAME,
                        help="gs://<bucket> or a local directory; reads <bucket>/features/")
    parser.add_argument("--mode", choices=["tune", "final"], required=True)
    # Tune mode tries every combination of the values given; final mode takes one value each.
    parser.add_argument("--max-depth", type=int, nargs="+", default=[5])
    parser.add_argument("--max-iter", type=int, nargs="+", default=[100])
    parser.add_argument("--step-size", type=float, nargs="+", default=[0.1])
    parser.add_argument("--max-bins", type=int, nargs="+", default=[128],
                        help="split candidates per feature; Spark's 32 would put the rare 10-30 mm thresholds "
                             "~3 mm apart")
    parser.add_argument("--min-instances", type=int, nargs="+", default=[100],
                        help="fewest training rows per leaf, so deep trees can't memorise single storms")
    parser.add_argument("--drop", nargs="*", default=[], choices=list(FEATURE_GROUPS),
                        help="feature groups to leave out (ablation)")
    parser.add_argument("--sample-days", type=float, default=1.0,
                        help="share of training days used to fit the GBT, to save cluster time")
    args = parser.parse_args()
    if not args.bucket:
        parser.error("pass --bucket or set BUCKET_NAME")
    if args.mode == "final" and any(len(getattr(args, a)) > 1 for _, a, _ in GBT_PARAMS):
        parser.error("final mode takes one value per GBT setting")
    root = args.bucket.rstrip("/")
    if "://" not in root and not os.path.isdir(root):
        root = "gs://" + root

    spark = (SparkSession.builder
             .appName("Train Model")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())
    # GBT checkpoints every 10 iterations (checkpointInterval) only if this is set; without it the lineage
    # grows with every tree and ~200 iterations on the full data overflow the executors' stack.
    # A relative path lands on the cluster's HDFS, which is deleted with the cluster.
    spark.sparkContext.setCheckpointDir("checkpoints")

    features = spark.read.parquet(f"{root}/features")
    if args.mode == "tune":
        tune(spark, features.where(F.col("year").isin(TRAIN_YEARS)), args, root)
    else:
        final(spark, features, args, root)


if __name__ == "__main__":
    main()
