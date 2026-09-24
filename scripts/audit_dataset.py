"""
Audit the raw rainfall CSVs (one file per year) and write the results to <bucket>/audit/.

The checks follow the audit checklist. Every output is a folder holding one CSV with a header:

  summary/             one row per year: every count below, side by side
  invalid_rows/        every row that fails a row-level check, with its reasons and severity
  station_year/        one row per station and year: coverage, positions, stuck-gauge and level-shift signs
  station_positions/   every position each station has had, and how far it is from the first
  name_id_conflicts/   names used by several IDs and IDs with several names
  timestamp_patterns/  rows per month by (minute mod 5, seconds)
  update_lag/          time from the reading to its last update
  gaps/                periods over 10 minutes with no station reporting
  daily_stations/      stations reporting per 5-minute bucket, per day
  hourly_profile/      rain by hour of day, to confirm local time
  monthly_island/      island-mean monthly rainfall

Severity in invalid_rows:
  drop    the row is removed by the clean step
  fix     the row is kept after a mechanical fix (value snapped to the 0.2 mm grid)
  review  the row is kept, but a person should look at it

Run on Dataproc (Spark reads gs:// directly):
  gcloud dataproc jobs submit pyspark scripts/audit_dataset.py \
      --cluster=<cluster> --region=asia-southeast1 -- --bucket gs://<bucket>

"""

import argparse
import os
from functools import reduce

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window

from utils import YEARS, SG_LAT, SG_LON, BUCKET_NAME


COLUMNS = ["date", "timestamp", "update_timestamp", "station_id", "station_name", "station_device_id",
           "location_longitude", "location_latitude", "reading_update_timestamp", "reading_value",
           "reading_type", "reading_unit"]
EXPECTED_TYPE = "TB1 Rainfall 5 Minute Total F"
BUCKET_S = 300                          # 5 minutes
EXTREME_MM = 20.0                       # 5-minute totals above this are flagged for review
MIN_COVERAGE = 0.10                     # stations below this share of a year's buckets are not used that year
FULL_DAY = 230                          # a station-day counts as complete with >= 230 of 288 readings
WET_DAY_SHARE = 0.75                    # a day is "wet island-wide" when >= 75% of complete stations get >= 1 mm
NEW_SITE_M = 500                        # a move further than this is treated as a new site

SEVERITY = {
    # reason                          severity
    "malformed_line":                 "drop",  
    "empty_cell":                     "drop",  
    "timestamp_unparseable":          "drop",  
    "timezone_not_sgt":               "drop",  
    "timestamp_off_grid":             "drop",  
    "unit_not_mm":                    "drop",  
    "value_unparseable":              "drop",  
    "value_negative":                 "drop",  
    "duplicate_dropped":              "drop",
    "value_off_grid":                 "fix",   
    "date_mismatch":                  "review",
    "outside_file_year":              "review",
    "duplicate_value_conflict":       "review",
    "duplicate_across_files":         "review",
    "update_before_reading_update":   "review",
    "reading_update_before_reading":  "review",
    "device_id_differs":              "review",
    "position_outside_sg":            "review",
    "value_extreme":                  "review",
    "type_unexpected":                "review",
}


def reasons_with(severity):
    return F.array(*[F.lit(r) for r, s in SEVERITY.items() if s == severity])


def reasons_array(checks):
    """Names of the checks whose condition is true (a null condition counts as false)."""
    flags = [F.when(F.coalesce(cond, F.lit(False)), F.lit(name)) for name, cond in checks]
    return F.filter(F.array(*flags), lambda x: x.isNotNull())


def distance_m(lat1, lon1, lat2, lon2):
    """Equirectangular approximation; accurate to well under 1% at Singapore's scale."""
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * F.cos(F.radians((lat1 + lat2) / 2))
    return F.round(111_320 * F.sqrt(dlat * dlat + dlon * dlon))


def read_year(spark, raw_dir, year):
    """Every column as a string, plus _corrupt_record for lines that don't fit the 12 columns."""
    path = f"{raw_dir}/rainfall_across_sg_{year}.csv"
    header = spark.read.text(path).first()[0].lstrip("\ufeff").strip().split(",")
    schema = T.StructType([T.StructField(c, T.StringType()) for c in COLUMNS]
                          + [T.StructField("_corrupt_record", T.StringType())])
    df = (spark.read.csv(path, header=True, schema=schema, mode="PERMISSIVE",
                         columnNameOfCorruptRecord="_corrupt_record")
          .withColumn("file_year", F.lit(year)))
    return df, header == COLUMNS


def parse(df):
    return (df
            .withColumn("ts", F.to_timestamp("timestamp"))
            .withColumn("reading_upd", F.to_timestamp("reading_update_timestamp"))
            .withColumn("upd", F.to_timestamp("update_timestamp"))
            # Timestamps mark the END of the interval; round up so 11:19:59 and 11:20:00 share a bucket.
            .withColumn("bucket", F.expr(f"timestamp_seconds(ceil(unix_seconds(ts) / {BUCKET_S}) * {BUCKET_S})"))
            .withColumn("start", F.col("bucket") - F.expr("INTERVAL 5 MINUTES"))
            .withColumn("v", F.col("reading_value").cast("double"))
            .withColumn("lat", F.col("location_latitude").cast("double"))
            .withColumn("lon", F.col("location_longitude").cast("double")))


def flag_rows(df):
    minute = F.substring("timestamp", 15, 2).cast("int")
    second = F.substring("timestamp", 18, 2)
    empty = reduce(lambda a, b: a | b, [F.col(c).isNull() | (F.trim(F.col(c)) == "") for c in COLUMNS])
    in_sg = F.col("lat").between(*SG_LAT) & F.col("lon").between(*SG_LON)

    row_checks = [
        ("malformed_line",                F.col("_corrupt_record").isNotNull()),
        ("empty_cell",                    empty),
        ("timestamp_unparseable",         F.col("timestamp").isNotNull() & F.col("ts").isNull()),
        ("timezone_not_sgt",              F.substring("timestamp", 20, 6) != "+08:00"),
        ("timestamp_off_grid",            ~(((minute % 5 == 4) & (second == "59"))
                                            | ((minute % 5 == 0) & (second == "00")))),
        ("date_mismatch",                 F.col("date") != F.substring("timestamp", 1, 10)),
        ("outside_file_year",             F.year("start") != F.col("file_year")),
        ("update_before_reading_update",  F.col("upd") < F.col("reading_upd")),
        ("reading_update_before_reading", F.col("reading_upd") < F.col("ts")),
        ("device_id_differs",             F.col("station_device_id") != F.col("station_id")),
        ("position_outside_sg",           F.col("lat").isNull() | F.col("lon").isNull() | ~in_sg),
        ("type_unexpected",               F.col("reading_type") != EXPECTED_TYPE),
        ("unit_not_mm",                   F.col("reading_unit") != "mm"),
        ("value_unparseable",             F.col("reading_value").isNotNull() & F.col("v").isNull()),
        ("value_negative",                F.col("v") < 0),
        ("value_off_grid",                F.abs(F.col("v") * 5 - F.round(F.col("v") * 5)) > 1e-6),
        ("value_extreme",                 F.col("v") > EXTREME_MM),
    ]
    df = (df.withColumn("base_reasons", reasons_array(row_checks))
            .withColumn("base_drop", F.arrays_overlap("base_reasons", reasons_with("drop"))))

    # Duplicates are judged among rows that survive the row checks; dropped rows form their own groups.
    group = Window.partitionBy("station_id", "bucket", "base_drop")
    newest_first = group.orderBy(F.col("reading_upd").desc_nulls_last(), F.col("upd").desc_nulls_last(),
                                 F.col("file_year").desc(), F.col("v").desc_nulls_last())
    df = (df.withColumn("dup_rank", F.row_number().over(newest_first))
            .withColumn("dup_values", F.size(F.collect_set("v").over(group)))
            .withColumn("dup_files", F.size(F.collect_set("file_year").over(group)))
            .withColumn("raw_ts_copies", F.count(F.lit(1)).over(Window.partitionBy("station_id", "timestamp"))))
    kept = ~F.col("base_drop")
    dup_checks = [
        ("duplicate_dropped",        kept & (F.col("dup_rank") > 1)),
        ("duplicate_value_conflict", kept & (F.col("dup_values") > 1)),
        ("duplicate_across_files",   kept & (F.col("dup_files") > 1)),
    ]
    df = df.withColumn("reasons", F.concat("base_reasons", reasons_array(dup_checks)))
    return df.withColumn(
        "severity",
        F.when(F.size("reasons") == 0, F.lit(None).cast("string"))
         .when(F.arrays_overlap("reasons", reasons_with("drop")), "drop")
         .when(F.arrays_overlap("reasons", reasons_with("fix")), "fix")
         .otherwise("review"))


def station_year_table(clean, year_buckets):
    base = (clean.groupBy("file_year", "station_id")
            .agg(F.count(F.lit(1)).alias("rows"),
                 F.min("bucket").alias("first_bucket"),
                 F.max("bucket").alias("last_bucket"),
                 F.array_join(F.array_sort(F.collect_set("station_name")), " | ").alias("names"),
                 F.countDistinct("lat", "lon").alias("positions"),
                 F.avg("v").alias("mean_mm_per_reading"))
            .join(year_buckets.select("file_year", "buckets", "expected_buckets"), "file_year")
            .withColumn("coverage", F.round(F.col("rows") / F.col("buckets"), 4))
            .withColumn("below_min_coverage", F.col("coverage") < MIN_COVERAGE)
            .withColumn("annual_total_est_mm", F.round(F.col("mean_mm_per_reading") * F.col("expected_buckets"), 1)))

    # Level shift = station mean compared with the median station that year (stations with enough coverage).
    medians = (base.where(~F.col("below_min_coverage")).groupBy("file_year")
               .agg(F.percentile_approx("mean_mm_per_reading", 0.5).alias("median_mean_mm")))
    base = (base.join(medians, "file_year", "left")
                .withColumn("ratio_to_median_station",
                            F.round(F.col("mean_mm_per_reading") / F.col("median_mean_mm"), 3)))

    # Dry on island-wide wet days = a complete station-day with 0 mm while >= 75% of complete stations got >= 1 mm.
    days = (clean.withColumn("day", F.to_date("start"))
            .groupBy("file_year", "station_id", "day")
            .agg(F.count(F.lit(1)).alias("n"), F.sum("v").alias("total"))
            .where(F.col("n") >= FULL_DAY)
            .withColumn("island_wet_share",
                        F.avg((F.col("total") >= 1).cast("double")).over(Window.partitionBy("file_year", "day"))))
    wet_days = (days.where(F.col("island_wet_share") >= WET_DAY_SHARE)
                .groupBy("file_year", "station_id")
                .agg(F.count(F.lit(1)).alias("island_wet_days_reported"),
                     F.sum((F.col("total") == 0).cast("int")).alias("dry_on_island_wet_days")))

    # Stuck gauge = the longest run of the same non-zero value in consecutive buckets.
    w = Window.partitionBy("station_id").orderBy("bucket")
    runs = (clean.select("file_year", "station_id", "bucket", "v")
            .withColumn("prev_v", F.lag("v").over(w))
            .withColumn("prev_b", F.lag("bucket").over(w))
            .withColumn("new_run", (F.col("v") == 0) | F.col("prev_v").isNull() | (F.col("v") != F.col("prev_v"))
                        | (F.unix_timestamp("bucket") - F.unix_timestamp("prev_b") != BUCKET_S))
            .withColumn("run_id", F.sum(F.col("new_run").cast("int")).over(w.rowsBetween(Window.unboundedPreceding, 0)))
            .where(F.col("v") > 0)
            .groupBy("file_year", "station_id", "run_id").agg(F.count(F.lit(1)).alias("run_len"))
            .groupBy("file_year", "station_id").agg(F.max("run_len").alias("longest_same_nonzero_run")))

    return (base.join(wet_days, ["file_year", "station_id"], "left")
                .join(runs, ["file_year", "station_id"], "left")
                .fillna(0, ["island_wet_days_reported", "dry_on_island_wet_days", "longest_same_nonzero_run"])
                .select("file_year", "station_id", "names", "rows", "coverage", "below_min_coverage",
                        "first_bucket", "last_bucket", "positions", "annual_total_est_mm",
                        "ratio_to_median_station", "island_wet_days_reported", "dry_on_island_wet_days",
                        "longest_same_nonzero_run")
                .orderBy("file_year", "coverage"))


def main():
    parser = argparse.ArgumentParser(description="Audit the raw rainfall CSVs.")
    parser.add_argument("--bucket", default=BUCKET_NAME,
                        help="gs://<bucket> or a local directory; raw files are read from <bucket>/raw/")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("pass --bucket or set BUCKET_NAME")
    root = args.bucket.rstrip("/")
    if "://" not in root and not os.path.isdir(root):
        root = "gs://" + root
    raw_dir, out_dir = f"{root}/raw", f"{root}/audit"

    spark = (SparkSession.builder
             .appName("Audit Dataset")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())

    def write(df, name):
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/{name}")
        print(f"wrote {out_dir}/{name}")

    # A1 + read
    frames, header_ok = [], {}
    for year in args.years:
        df, ok = read_year(spark, raw_dir, year)
        frames.append(df)
        header_ok[year] = ok
        print(f"{year}: header {'OK' if ok else 'DIFFERS from 2017'}")
    raw = reduce(lambda a, b: a.unionByName(b), frames)

    rows = flag_rows(parse(raw)).persist(StorageLevel.MEMORY_AND_DISK)

    # Rows the clean step would keep, with values snapped to the 0.2 mm grid.
    clean = (rows.where(F.coalesce(F.col("severity") != "drop", F.lit(True)))
             .select("file_year", "station_id", "station_name", "lat", "lon", "bucket", "start",
                     (F.round(F.col("v") * 5) / 5).alias("v"))
             .persist(StorageLevel.MEMORY_AND_DISK))

    per_bucket = (clean.groupBy("file_year", "bucket", "start")
                  .agg(F.count(F.lit(1)).alias("stations"),
                       F.sum((F.col("v") > 0).cast("int")).alias("wet_stations"))
                  .persist(StorageLevel.MEMORY_AND_DISK))

    year_buckets = (per_bucket.groupBy("file_year")
                    .agg(F.count(F.lit(1)).alias("buckets"),
                         F.min("bucket").alias("first_bucket"),
                         F.max("bucket").alias("last_bucket"))
                    .withColumn("expected_buckets",
                                F.expr("datediff(make_date(file_year + 1, 1, 1), make_date(file_year, 1, 1)) * 288")))

    # ---- invalid rows (with how many stations were wet in the same bucket, to judge extreme values)
    invalid = (rows.where(F.size("reasons") > 0)
               .join(per_bucket.select("file_year", "bucket", "stations", "wet_stations"),
                     ["file_year", "bucket"], "left")
               .select("file_year", "severity", F.array_join("reasons", ";").alias("reasons"), *COLUMNS,
                       "bucket", F.col("stations").alias("stations_in_bucket"),
                       F.col("wet_stations").alias("wet_stations_in_bucket"), "_corrupt_record")
               .orderBy("file_year", "timestamp", "station_id"))
    write(invalid, "invalid_rows")

    # ---- Timestamp patterns
    write(rows.groupBy("file_year", F.substring("timestamp", 1, 7).alias("month"),
                       (F.substring("timestamp", 15, 2).cast("int") % 5).alias("minute_mod_5"),
                       F.substring("timestamp", 18, 2).alias("seconds"))
              .agg(F.count(F.lit(1)).alias("rows"), F.min("timestamp").alias("first"),
                   F.max("timestamp").alias("last"))
              .orderBy("file_year", "month", "minute_mod_5", "seconds"),
          "timestamp_patterns")

    # ---- Update lag
    lag_s = F.unix_timestamp("reading_upd") - F.unix_timestamp("ts")
    write(rows.where(~F.col("base_drop"))
              .withColumn("reading_update_lag",
                          F.when(lag_s < 0, "0. negative").when(lag_s < 900, "1. < 15 min")
                           .when(lag_s < 3600, "2. < 1 h").when(lag_s < 86400, "3. < 1 day")
                           .otherwise("4. >= 1 day"))
              .groupBy("file_year", "reading_update_lag").count()
              .orderBy("file_year", "reading_update_lag"),
          "update_lag")

    # ---- name / ID conflicts
    names_per_id = (clean.groupBy("file_year", "station_id")
                    .agg(F.array_join(F.array_sort(F.collect_set("station_name")), " | ").alias("values"),
                         F.countDistinct("station_name").alias("n"))
                    .where(F.col("n") > 1)
                    .select("file_year", F.lit("id_with_several_names").alias("kind"),
                            F.col("station_id").alias("key"), "values"))
    ids_per_name = (clean.groupBy("file_year", "station_name")
                    .agg(F.array_join(F.array_sort(F.collect_set("station_id")), " | ").alias("values"),
                         F.countDistinct("station_id").alias("n"))
                    .where(F.col("n") > 1)
                    .select("file_year", F.lit("name_with_several_ids").alias("kind"),
                            F.col("station_name").alias("key"), "values"))
    name_conflicts = names_per_id.unionByName(ids_per_name).persist()
    write(name_conflicts.orderBy("file_year", "kind", "key"), "name_id_conflicts")

    # ---- Station positions
    positions = (clean.groupBy("station_id", "lat", "lon")
                 .agg(F.min("bucket").alias("first_bucket"), F.max("bucket").alias("last_bucket"),
                      F.count(F.lit(1)).alias("rows"),
                      F.array_join(F.array_sort(F.collect_set(F.col("file_year").cast("string"))), " ").alias("years")))
    first_pos = Window.partitionBy("station_id").orderBy("first_bucket")
    positions = (positions
                 .withColumn("first_lat", F.first("lat").over(first_pos))
                 .withColumn("first_lon", F.first("lon").over(first_pos))
                 .withColumn("distance_from_first_m", distance_m(F.col("first_lat"), F.col("first_lon"),
                                                                 F.col("lat"), F.col("lon")))
                 .withColumn("positions", F.count(F.lit(1)).over(Window.partitionBy("station_id")))
                 .withColumn("new_site", F.col("distance_from_first_m") > NEW_SITE_M)
                 .drop("first_lat", "first_lon"))
    write(positions.where(F.col("positions") > 1).orderBy("station_id", "first_bucket"), "station_positions")

    # ---- station-year table
    station_year = station_year_table(clean, year_buckets).persist()
    write(station_year, "station_year")

    # ---- gaps with no station reporting
    by_bucket = Window.partitionBy("file_year").orderBy("bucket")
    gaps = (per_bucket.select("file_year", "bucket")
            .withColumn("prev_bucket", F.lag("bucket").over(by_bucket))
            .withColumn("gap_min", (F.unix_timestamp("bucket") - F.unix_timestamp("prev_bucket")) / 60)
            .where(F.col("gap_min") > 10)
            .select("file_year", F.col("prev_bucket").alias("last_before_gap"),
                    F.col("bucket").alias("first_after_gap"), "gap_min")
            .persist())
    write(gaps.orderBy("file_year", "last_before_gap"), "gaps")

    # ---- stations reporting per bucket, per day
    write(per_bucket.groupBy("file_year", F.to_date("start").alias("day"))
                    .agg(F.count(F.lit(1)).alias("buckets"),
                         F.min("stations").alias("min_stations"),
                         F.round(F.avg("stations"), 1).alias("mean_stations"),
                         F.sum((F.col("stations") < 35).cast("int")).alias("buckets_under_35"))
                    .orderBy("file_year", "day"),
          "daily_stations")

    # ---- hourly profile (by interval start, local time)
    write(clean.groupBy("file_year", F.hour("start").alias("hour"))
               .agg(F.round(F.avg("v"), 4).alias("mean_mm_per_reading"),
                    F.round(F.avg((F.col("v") > 0).cast("double")), 4).alias("wet_share"))
               .orderBy("file_year", "hour"),
          "hourly_profile")

    # ---- island-mean monthly rainfall (mean per reading x readings in a full month)
    write(clean.groupBy("file_year", F.month("start").alias("month"))
               .agg(F.avg("v").alias("mean_mm"), F.first(F.dayofmonth(F.last_day("start"))).alias("days"))
               .select("file_year", "month",
                       F.round(F.col("mean_mm") * 288 * F.col("days"), 1).alias("island_mean_total_mm"))
               .orderBy("file_year", "month"),
          "monthly_island")

    # ---- summary: one row per year
    reason_counts = [F.sum(F.array_contains("reasons", r).cast("int")).alias(r) for r in SEVERITY]
    row_stats = (rows.groupBy("file_year").agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum((F.col("severity") == "drop").cast("int")).alias("rows_drop"),
        F.sum((F.col("severity") == "fix").cast("int")).alias("rows_fix"),
        F.sum((F.col("severity") == "review").cast("int")).alias("rows_review"),
        F.sum((F.col("raw_ts_copies") > 1).cast("int")).alias("rows_sharing_raw_timestamp"),
        F.round(F.avg((F.col("upd") == F.col("reading_upd")).cast("double")), 4).alias("update_cols_equal_share"),
        F.array_join(F.collect_set("reading_type"), " | ").alias("reading_types"),
        F.array_join(F.collect_set("reading_unit"), " | ").alias("reading_units"),
        F.min(F.when(F.col("v") > 0, F.col("v"))).alias("min_nonzero_mm"),
        F.max("v").alias("max_mm"),
        F.round(F.avg((F.col("v") == 0).cast("double")), 4).alias("zero_share"),
        *reason_counts))
    bucket_stats = (per_bucket.groupBy("file_year").agg(
        F.sum((F.col("stations") == 1).cast("int")).alias("one_station_buckets"),
        F.countDistinct(F.when(F.col("stations") < 35, F.to_date("start"))).alias("days_with_bucket_under_35")))
    clean_stats = clean.groupBy("file_year").agg(F.countDistinct("station_id").alias("stations"),
                                                 F.avg("v").alias("mean_mm"))
    gap_stats = gaps.groupBy("file_year").agg(F.count(F.lit(1)).alias("gaps_over_10_min"),
                                              F.max("gap_min").alias("longest_gap_min"))
    sy_stats = station_year.groupBy("file_year").agg(
        F.sum(F.col("below_min_coverage").cast("int")).alias("stations_below_min_coverage"),
        F.sum((F.col("positions") > 1).cast("int")).alias("stations_moved_in_year"),
        F.sum((F.col("dry_on_island_wet_days") > 0).cast("int")).alias("stations_dry_on_wet_days"))
    conflict_stats = name_conflicts.groupBy("file_year").agg(F.count(F.lit(1)).alias("name_id_conflicts"))

    summary = (row_stats
               .join(year_buckets, "file_year", "left")
               .join(bucket_stats, "file_year", "left")
               .join(clean_stats, "file_year", "left")
               .join(gap_stats, "file_year", "left")
               .join(sy_stats, "file_year", "left")
               .join(conflict_stats, "file_year", "left")
               .withColumn("bucket_coverage", F.round(F.col("buckets") / F.col("expected_buckets"), 4))
               .withColumn("island_total_est_mm", F.round(F.col("mean_mm") * F.col("expected_buckets"), 1))
               .drop("mean_mm")
               .fillna(0, ["gaps_over_10_min", "name_id_conflicts"]))

    # Stations new or gone compared with the previous year; A1: header check.
    station_sets = {r["file_year"]: set(r["ids"]) for r in
                    station_year.groupBy("file_year").agg(F.collect_set("station_id").alias("ids")).collect()}
    extra = []
    for year in args.years:
        now, before = station_sets.get(year, set()), station_sets.get(year - 1)
        new = " ".join(sorted(now - before)) if before is not None else ""
        gone = " ".join(sorted(before - now)) if before is not None else ""
        extra.append((year, header_ok[year], new, gone))
    extra_df = spark.createDataFrame(extra, "file_year int, header_ok boolean, stations_new string, stations_gone string")
    summary = extra_df.join(summary, "file_year", "left").orderBy("file_year")
    write(summary, "summary")

    summary.show(truncate=False, vertical=True)
    spark.stop()


if __name__ == "__main__":
    main()
