"""
Build the Task 2 feature and label rows from clean/ (plan.md §3, step 3; features §4.5, missing data §4.6).

One row per prediction: a station that is active that year (coverage >= MIN_COVERAGE), has a reading at t,
and has all six readings in (t, t + 30 min]. All years are built in one run, so look-back windows and label
windows cross year boundaries; plan.md §4.3's one-day trim at split boundaries is the training step's job.

Outputs:
  features/            Parquet partitioned by year (the year the interval ending at t starts in):
                       station_id, bucket (= t), every feature in FEATURE_GROUPS, rain_next_30m
                       (train.py turns rain_next_30m into labels for each threshold)
  stations/            one CSV: per station and year, coverage, whether it is used, position, neighbours,
                       and the share of prediction rows dropped for an incomplete label window

Climatology is not in features/: it must come from each fold's training years only (plan.md §4.4), so the
training step adds it (train.py: climatology).

Run:
  scripts/run_script.sh build_features                        # all years
  scripts/run_script.sh build_features -- --years 2017 2018   # a cheap test
"""

import math
import argparse
import os

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window

from utils import YEARS, BUCKET_NAME, BUCKET_S, MIN_COVERAGE

EARTH_RADIUS_KM = 6371.0

# Look-back windows in 5-minute readings. A reading at t covers (t - 5 min, t], so n readings cover the last 5n min.
RAIN_WINDOWS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "3h": 36, "6h": 72}
COMPLETENESS_WINDOWS = {"1h": 12, "3h": 36, "6h": 72}
WET_CAP_MIN = 7 * 24 * 60               # time since last wet reading is capped at 7 days

LABEL_READINGS = 6                      # readings in the label window
K_NEIGHBOURS = 5
DIRECTIONS = ["n", "e", "s", "w"]
NO_STATION_KM = -1.0                    # direction distance when no station in that direction reports at t


STATION_FEATURES = ([f"rain_{k}" for k in RAIN_WINDOWS] + ["trend_15m", "min_since_wet"]
                    + [f"present_{k}" for k in COMPLETENESS_WINDOWS])

FEATURE_GROUPS = {
    "station": STATION_FEATURES,
    "neighbour": ["nbr_mean_30m", "nbr_max_30m", "nbr_count"],
    "direction": [f"dir_{d}_{x}" for d in DIRECTIONS for x in ("rain_30m", "km")],
    "island": ["island_mean_30m", "island_max_30m", "island_wet_share", "island_count"],
    "calendar": ["tod_sin", "tod_cos", "hour", "month"],
    "day_of_year": ["doy_sin", "doy_cos"],
    "place": ["lat", "lon"],
    "climatology": ["clim"],            # not in features/: train.py adds it per fold and threshold
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, as a Column expression."""
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (F.sin(dlat / 2) ** 2
         + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * F.asin(F.sqrt(a))


def offset_km(lat1, lon1, lat2, lon2):
    """(north, east) offset of point 2 from point 1; flat-earth, fine at Singapore's scale."""
    north = F.radians(lat2 - lat1) * EARTH_RADIUS_KM
    east = F.radians(lon2 - lon1) * F.cos(F.radians((lat1 + lat2) / 2)) * EARTH_RADIUS_KM
    return north, east


def cyclic(col, period):
    """(sin, cos) of a periodic value, so the end of the cycle sits next to the start.
    `period` can be a number or a Column (e.g. days in that year)."""
    angle = 2 * math.pi * col / period
    return F.sin(angle), F.cos(angle)


def snap(col):
    """Back onto the 0.2 mm grid: adding 0.2s in floating point drifts (0.2 * 3 != 0.6)."""
    return F.round(col * 5) / 5


def station_grid(clean):
    """Every 5-minute bucket between each station's first and last reading (plan.md §4.6).
    value_mm is null where the reading is missing, so gaps can't pass for dry spells."""
    spans = clean.groupBy("station_id").agg(F.min("bucket").alias("first"), F.max("bucket").alias("last"))
    grid = spans.select("station_id",
                        F.explode(F.sequence("first", "last", F.expr(f"INTERVAL {BUCKET_S} SECONDS")))
                         .alias("bucket"))
    return (grid.join(clean.select("station_id", "bucket", "value_mm"), ["station_id", "bucket"], "left")
                # Same rule as clean/: the year the interval STARTS in.
                .withColumn("year", F.year(F.col("bucket") - F.expr(f"INTERVAL {BUCKET_S} SECONDS"))))


def station_features(grid):
    """The station group of plan.md §4.5, computed at every grid bucket t from readings at or before t.

    Windows count rows, not time, so `grid` must have exactly one row per station and 5-minute bucket
    (station_grid). Rain sums add the readings present; present_* says what share of the window was seen.
    Filter to prediction rows (value_mm not null at t) only after this, or the windows lose their gaps.
    """
    w = Window.partitionBy("station_id").orderBy("bucket")

    def last(n, skip=0):
        """The n readings ending `skip` readings before t."""
        return w.rowsBetween(-(n + skip - 1), -skip)

    def rain(n, skip=0):
        return F.coalesce(snap(F.sum("value_mm").over(last(n, skip))), F.lit(0.0))

    cols = {f"rain_{k}": rain(n) for k, n in RAIN_WINDOWS.items()}
    cols["trend_15m"] = snap(rain(3) - rain(3, skip=3))

    # A gap counts as "not wet": only readings that are present and > 0 move the last-wet time.
    last_wet = F.last(F.when(F.col("value_mm") > 0, F.col("bucket")), ignorenulls=True) \
                .over(w.rowsBetween(Window.unboundedPreceding, 0))
    since = (F.unix_timestamp("bucket") - F.unix_timestamp(last_wet)) / 60
    cols["min_since_wet"] = F.least(F.coalesce(since, F.lit(WET_CAP_MIN)), F.lit(WET_CAP_MIN))

    for k, n in COMPLETENESS_WINDOWS.items():
        cols[f"present_{k}"] = F.count("value_mm").over(last(n)) / n

    return grid.withColumns(cols)


def labels(grid):
    """Rain in (t, t + 30 min], the quantity every label is cut from (plan.md §4.1). Same row-count rule as
    station_features. next_present < LABEL_READINGS means the window has a gap and the row must be dropped."""
    ahead = Window.partitionBy("station_id").orderBy("bucket").rowsBetween(1, LABEL_READINGS)
    # Snapped, or a sum of 0.2s can land just off a threshold.
    return grid.withColumns({"rain_next_30m": snap(F.sum("value_mm").over(ahead)),
                             "next_present": F.count("value_mm").over(ahead)})


def station_year(clean):
    """Per station and year: coverage (share of the year's buckets it reports), whether it meets
    MIN_COVERAGE, and its latest position that year (plan.md §2, §4.6)."""
    buckets = clean.groupBy("year").agg(F.countDistinct("bucket").alias("year_buckets"))
    return (clean.groupBy("station_id", "year")
            .agg(F.count(F.lit(1)).alias("readings"),
                 F.max_by(F.struct("lat", "lon"), "bucket").alias("pos"))
            .join(buckets, "year")
            .select("station_id", "year", "readings",
                    F.round(F.col("readings") / F.col("year_buckets"), 4).alias("coverage"),
                    (F.col("readings") / F.col("year_buckets") >= MIN_COVERAGE).alias("active"),
                    F.col("pos.lat").alias("lat"), F.col("pos.lon").alias("lon")))


def neighbour_lists(stations):
    """Per active station and year: the K_NEIGHBOURS nearest other active stations, and every other active
    station in each compass quadrant, nearest first. Each list holds (km, nbr_id) structs. Chosen once per
    year so they never point at a station that isn't running that year (plan.md §4.5)."""
    active = stations.where("active").select("station_id", "year", "lat", "lon")
    other = active.select(F.col("station_id").alias("nbr_id"), "year",
                          F.col("lat").alias("nbr_lat"), F.col("lon").alias("nbr_lon"))
    north, east = offset_km(F.col("lat"), F.col("lon"), F.col("nbr_lat"), F.col("nbr_lon"))
    pairs = (active.join(other, "year")
             .where(F.col("station_id") != F.col("nbr_id"))
             .select("station_id", "year", "nbr_id",
                     haversine_km(F.col("lat"), F.col("lon"), F.col("nbr_lat"), F.col("nbr_lon")).alias("km"),
                     F.when(F.abs(north) >= F.abs(east), F.when(north > 0, "n").otherwise("s"))
                      .otherwise(F.when(east > 0, "e").otherwise("w")).alias("direction")))
    entry = F.struct("km", "nbr_id")    # array_sort orders structs by their first field, km
    return (pairs.groupBy("station_id", "year")
            .agg(F.slice(F.array_sort(F.collect_list(entry)), 1, K_NEIGHBOURS).alias("nearest"),
                 *[F.array_sort(F.collect_list(F.when(F.col("direction") == d, entry))).alias(f"dir_{d}")
                   for d in DIRECTIONS]))


def island_snapshot(reporting):
    """One row per bucket from the active stations reporting at t: their 30-min rain as a
    station_id -> rain_30m map (for neighbour lookups), plus the island group of plan.md §4.5."""
    return (reporting.groupBy("bucket")
            .agg(F.map_from_entries(F.collect_list(F.struct("station_id", "rain_30m"))).alias("rain30"),
                 F.avg("rain_30m").alias("island_mean_30m"),
                 F.max("rain_30m").alias("island_max_30m"),
                 F.avg((F.col("rain_30m") > 0).cast("double")).alias("island_wet_share"),
                 F.count(F.lit(1)).alias("island_count")))


def network_features(rows):
    """Neighbour and direction groups of plan.md §4.5. `rows` carries rain30 (island_snapshot) and the
    lists from neighbour_lists. Only stations reporting at t count; when none do, rain is 0, the count
    says so for neighbours, and the distance is NO_STATION_KM for directions (plan.md §4.6)."""
    def rain_at_t(nbr_id):
        return F.try_element_at(F.col("rain30"), nbr_id)   # null when that station isn't reporting at t

    nbr = F.filter(F.transform("nearest", lambda s: rain_at_t(s["nbr_id"])), lambda v: v.isNotNull())
    n = F.size(nbr)
    cols = {"nbr_count": n,
            "nbr_mean_30m": F.when(n > 0, F.aggregate(nbr, F.lit(0.0), lambda acc, v: acc + v) / n)
                             .otherwise(0.0),
            "nbr_max_30m": F.coalesce(F.array_max(nbr), F.lit(0.0))}

    for d in DIRECTIONS:
        hits = F.filter(F.transform(f"dir_{d}",
                                    lambda s: F.struct(rain_at_t(s["nbr_id"]).alias("rain"), s["km"].alias("km"))),
                        lambda s: s["rain"].isNotNull())
        first = F.try_element_at(hits, F.lit(1))
        cols[f"dir_{d}_rain_30m"] = F.coalesce(first["rain"], F.lit(0.0))
        cols[f"dir_{d}_km"] = F.coalesce(first["km"], F.lit(NO_STATION_KM))
    return rows.withColumns(cols)


def calendar_features(rows):
    """Calendar and day_of_year groups of plan.md §4.5, from t in Singapore time (the session time zone)."""
    tod_sin, tod_cos = cyclic(F.hour("bucket") * 60 + F.minute("bucket"), 24 * 60)
    days_in_year = F.dayofyear(F.expr("make_date(year(bucket), 12, 31)"))  # 365 or 366
    doy_sin, doy_cos = cyclic(F.dayofyear("bucket") - 1, days_in_year)
    return rows.withColumns({"tod_sin": tod_sin, "tod_cos": tod_cos,
                             "hour": F.hour("bucket"), "month": F.month("bucket"),
                             "doy_sin": doy_sin, "doy_cos": doy_cos})


def main():
    parser = argparse.ArgumentParser(description="Build Task 2 features and labels from clean/.")
    parser.add_argument("--bucket", default=BUCKET_NAME,
                        help="gs://<bucket> or a local directory; reads <bucket>/clean/")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("pass --bucket or set BUCKET_NAME")
    root = args.bucket.rstrip("/")
    if "://" not in root and not os.path.isdir(root):
        root = "gs://" + root

    spark = (SparkSession.builder
             .appName("Build Features")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())

    clean = spark.read.parquet(f"{root}/clean").where(F.col("year").isin(args.years))

    # Station features and labels share one window ordering, so Spark sorts the grid once.
    # ponytail: DISK_ONLY because ~50M wide rows in memory got the 64 GB node OOM-killed; MEMORY_AND_DISK on a bigger node
    grid = labels(station_features(station_grid(clean))).persist(StorageLevel.DISK_ONLY)
    stations = station_year(clean).persist(StorageLevel.MEMORY_AND_DISK)
    lists = neighbour_lists(stations)

    # Active stations with a reading at t: the targets, and the only stations neighbour/island features see.
    reporting = (grid.where(F.col("value_mm").isNotNull())
                 .join(F.broadcast(stations.where("active").select("station_id", "year", "lat", "lon")),
                       ["station_id", "year"]))
    snapshot = island_snapshot(reporting)

    rows = (reporting.where(F.col("next_present") == LABEL_READINGS)
            .join(snapshot, "bucket")
            .join(F.broadcast(lists), ["station_id", "year"]))
    rows = calendar_features(network_features(rows))

    out_cols = (["station_id", "year", "bucket"]
                + [c for g, cs in FEATURE_GROUPS.items() if g != "climatology" for c in cs]
                + ["rain_next_30m"])
    (rows.select(*out_cols)
         .repartition("year")
         .sortWithinPartitions("station_id", "bucket")
         .write.mode("overwrite")
         .partitionBy("year")
         .parquet(f"{root}/features"))
    print(f"wrote {root}/features")

    # Per station and year: which stations were used, and how many prediction rows the label rule dropped.
    readings = (grid.where(F.col("value_mm").isNotNull())
                .groupBy("station_id", "year")
                .agg(F.sum((F.col("next_present") == LABEL_READINGS).cast("int")).alias("label_complete")))
    names = F.array_join(F.transform("nearest", lambda s: s["nbr_id"]), " ")
    summary = (stations.join(readings, ["station_id", "year"], "left")
               .join(lists.select("station_id", "year", names.alias("neighbours")), ["station_id", "year"], "left")
               .withColumn("label_dropped_share",
                           F.round(1 - F.col("label_complete") / F.col("readings"), 4))
               .select("year", "station_id", "coverage", "active", "lat", "lon", "readings",
                       "label_complete", "label_dropped_share", "neighbours")
               .orderBy("year", "station_id"))
    summary.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{root}/stations")
    summary.where(~F.col("active")).show(100, truncate=False)
    print(f"wrote {root}/stations")


if __name__ == "__main__":
    main()
