"""
P(rain in the next 30 min > threshold mm) from the saved final model: one query, or the 2024 test evaluation.

Needs <bucket>/models/ from `train.py --mode final`, and the station's row at t in <bucket>/features/. That row
exists only when the station was active that year, reported at t and has the following 30 minutes complete, so
queries are for historical times. The time is rounded down to the 5-minute mark; the threshold is rounded down
to the 0.2 mm grid (exact, since readings come in 0.2 mm steps) and capped at 30 mm.

--evaluate is the test step (plan.md §4.7); it only loads models/, nothing is fitted. It scores 2023 to pick each
threshold's CSI cut-off and 2024 (both trimmed one day around 1 January) at EVAL_THRESHOLDS, and writes
  results/metrics/       per threshold and model (clim, pers, gbt, cal): Brier, BSS, ROC-AUC, PR-AUC, POD, FAR,
                         CSI, bias at the 2023 cut-off, ordering violations
  results/reliability/   calibrated model: predicted vs observed rate in 10 probability bins, per threshold
  results/predictions/   every 2024 row and threshold: rain_next_30m, label, clim, pers, gbt, cal

Run:
  scripts/run_script.sh predict -- --station S24 --time "2024-04-15 16:00" --threshold 5
  scripts/run_script.sh predict -- --evaluate
"""

import argparse
import os

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.sql import SparkSession

from train import (CALIB_YEAR, CLIM_PRIOR_WEIGHT, EVAL_THRESHOLDS, GRID_STEPS, MODELS, TEST_YEAR, TRAIN_YEARS,
                   add_arrays, at, best_cut, categorical, expand, grid_index, ordering_violations, summarise, trim,
                   write_csv, predict as gbt_probability)
from utils import BUCKET_NAME, BUCKET_S


def score(spark, root, rows, thresholds): 
    """rows (from features/) copied once per threshold, with clim, gbt (raw) and cal (calibrated probability)."""
    counts = spark.read.parquet(f"{root}/models/clim_counts")
    js = F.array(*[F.lit(grid_index(t)) for t in thresholds])
    df = gbt_probability(PipelineModel.load(f"{root}/models/gbt"), _climatology(expand(rows, js), counts, TRAIN_YEARS))
    return IsotonicRegressionModel.load(f"{root}/models/isotonic").transform(df)


def predict(spark, root, station_id, time, threshold):
    """Row with bucket (t used), threshold (mm used), gbt (raw) and cal (calibrated probability)."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0 mm")
    t = F.timestamp_seconds(F.floor(F.unix_timestamp(F.to_timestamp(F.lit(time))) / BUCKET_S) * BUCKET_S)
    row = spark.read.parquet(f"{root}/features").where((F.col("station_id") == station_id) & (F.col("bucket") == t))
    if row.isEmpty():
        raise LookupError(f"no features row for {station_id} at {time}: the station was inactive that year, "
                          "had no reading then, or the next 30 minutes are incomplete")
    return score(spark, root, row, [threshold]).select("bucket", "threshold", "gbt", "cal").first()

def evaluate(spark, root):
    """Test on 2024 data. Cut-offs come from 2023, as in final mode."""
    rows = trim(spark.read.parquet(f"{root}/features").where(F.col("year").isin(CALIB_YEAR, TEST_YEAR)),
                [CALIB_YEAR, TEST_YEAR])
    calib, test = [score(spark, root, rows.where(F.col("year") == y), EVAL_THRESHOLDS)
                   .persist(StorageLevel.MEMORY_AND_DISK) for y in (CALIB_YEAR, TEST_YEAR)]

    metrics, reliability = [], []
    broken = {"gbt": ordering_violations(test, "gbt"), "cal": ordering_violations(test, "cal")}
    for t in EVAL_THRESHOLDS:
        calib_t, test_t = at(calib, t), at(test, t)
        clim_brier = None
        # clim first: every model's Brier Skill Score is measured against it.
        for model in MODELS + ("cal",):
            cut = best_cut(summarise(calib_t, model)[1])      # cut-off chosen on 2023 only
            m, bins = summarise(test_t, model)
            clim_brier = clim_brier or m["brier"]
            metrics.append({"threshold": t, "model": model, **m, "bss": 1 - m["brier"] / clim_brier,
                            **categorical(bins, cut), "ordering_violations": broken.get(model)})
            if model == "cal":
                for d in range(10):
                    rs = [r for b, r in bins.items() if b // 10 == d]
                    n = sum(r.n for r in rs)
                    if n:
                        reliability.append({"threshold": t, "bin": f"{d / 10:.1f}-{(d + 1) / 10:.1f}", "n": n,
                                            "mean_p": sum(r.s for r in rs) / n,
                                            "observed": sum(r.k for r in rs) / n})
            print(metrics[-1])

    (test.select("station_id", "bucket", "threshold", "rain_next_30m", "label", "clim", "pers", "gbt", "cal")
         .write.mode("overwrite").parquet(f"{root}/results/predictions"))
    write_csv(spark, metrics, f"{root}/results/metrics")
    write_csv(spark, reliability, f"{root}/results/reliability")


def _climatology(rows, counts, train_years, weight=CLIM_PRIOR_WEIGHT):
    """train.climatology for rows outside the training years (their own-year counts are always 0),
    without element_at on a null array, which crashes Spark 4."""
    keys = ["station_id", "month", "hour"]
    tr = counts.where(F.col("year").isin(train_years))
    overall = tr.agg(F.sum("n").alias("n"), add_arrays(F.collect_list("k")).alias("k")).first()
    prior = F.array(*[F.lit(k / overall["n"]) for k in overall["k"]])
    total = tr.groupBy(*keys).agg(F.sum("n").alias("cell_n"), add_arrays(F.collect_list("k")).alias("cell_k"))
    i = F.col("j") + 1
    zero = F.array_repeat(F.lit(0).cast("long"), GRID_STEPS + 1)
    k = F.element_at(F.coalesce(F.col("cell_k"), zero), i)     # station/month/hour with no training data: 0
    n = F.coalesce(F.col("cell_n"), F.lit(0))
    return (rows.join(F.broadcast(total), keys, "left")
            .withColumn("clim", (k + weight * F.element_at(prior, i)) / (n + weight))
            .drop("cell_n", "cell_k"))


def main():
    parser = argparse.ArgumentParser(description="P(rain in the next 30 min > threshold) from the saved model.")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="gs://<bucket> or a local directory")
    parser.add_argument("--evaluate", action="store_true",
                        help="test on 2024: metrics, reliability and predictions in results/")
    parser.add_argument("--station", help="station_id for one query, e.g. S24")
    parser.add_argument("--time", help='Singapore time for one query, e.g. "2024-04-15 16:00"')
    parser.add_argument("--threshold", type=float, nargs="+", help="mm in the next 30 minutes")
    args = parser.parse_args()
    if not args.bucket:
        parser.error("pass --bucket or set BUCKET_NAME")
    if not args.evaluate and not (args.station and args.time and args.threshold):
        parser.error("pass --evaluate, or --station, --time and --threshold")
    root = args.bucket.rstrip("/")
    if "://" not in root and not os.path.isdir(root):
        root = "gs://" + root

    spark = (SparkSession.builder
             .appName("Predict")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())

    if args.evaluate:
        evaluate(spark, root)
        return

    for threshold in args.threshold:
        r = predict(spark, root, args.station, args.time, threshold)
        capped = " (capped: the grid ends at 30 mm)" if threshold > GRID_STEPS / 5 else ""
        print(f"P(more than {r.threshold:g} mm at {args.station} in the 30 min after {r.bucket}){capped} = "
              f"{r.cal:.3f}  (uncalibrated GBT: {r.gbt:.3f})")


if __name__ == "__main__":
    main()
