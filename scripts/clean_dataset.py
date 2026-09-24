"""
Clean the raw rainfall CSVs (plan.md §3, step 2) and write one Parquet dataset partitioned by year.

In this order: (1) round timestamps up to 5-minute buckets; (2) remove duplicates on (station, bucket),
keeping the latest reading_update_timestamp; (3) snap values to the 0.2 mm grid; (4) drop rows that fail
validation. The row checks and duplicate ranking are the ones in audit_dataset.py, so the rows dropped here
are exactly the rows audit/invalid_rows marks as "drop". Rows marked "review" are kept, with their reasons.

Outputs:
  clean/year=<year>/   one row per (station_id, bucket), sorted by station_id, bucket within each year
  clean_summary/       one CSV: per file year, rows read, dropped, kept, kept for review, kept in another year

Columns in clean/:
  station_id      the only station key
  station_name    display name, for labels only (not unique)
  lat, lon        position as reported in that row (can change, e.g. S113)
  bucket          end of the 5-minute interval, Singapore time; the reading's time everywhere downstream
  value_mm        rainfall over the interval, snapped to the 0.2 mm grid
  review          ";"-joined review reasons from the audit (e.g. value_extreme), null for most rows
  year            partition column: the year the interval STARTS in, so 2018-01-01 00:00 belongs to 2017

Clean all years in one run: a reading at 2017-12-31 23:59:59 rounds up to the same bucket as a 2018 row,
so duplicates across files are only found when both files are read together.

Run:
  scripts/run_script.sh clean_dataset                   # all years
  scripts/run_script.sh clean_dataset -- --years 2017   # one year, as a cheap test
"""

import argparse
import os
from functools import reduce

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import SparkSession

from audit_dataset import flag_rows, parse, read_year, reasons_with
from utils import YEARS, BUCKET_NAME


def main():
    parser = argparse.ArgumentParser(description="Clean the raw rainfall CSVs into a year-partitioned Parquet dataset.")
    parser.add_argument("--bucket", default=BUCKET_NAME,
                        help="gs://<bucket> or a local directory; raw files are read from <bucket>/raw/")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("pass --bucket or set BUCKET_NAME")
    root = args.bucket.rstrip("/")
    if "://" not in root and not os.path.isdir(root):
        root = "gs://" + root
    raw_dir = f"{root}/raw"

    spark = (SparkSession.builder
             .appName("Clean Dataset")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())

    frames = []
    for year in args.years:
        df, header_ok = read_year(spark, raw_dir, year)
        if not header_ok:
            print(f"{year}: header DIFFERS from 2017; check audit/summary before trusting this year")
        frames.append(df)
    raw = reduce(lambda a, b: a.unionByName(b), frames)

    # parse() rounds timestamps up to the bucket; flag_rows() applies the row checks, then ranks duplicates
    # per (station_id, bucket) by latest reading_update_timestamp and flags all but the first as dropped.
    rows = flag_rows(parse(raw)).persist(StorageLevel.MEMORY_AND_DISK)
    keep = F.coalesce(F.col("severity") != "drop", F.lit(True))

    review = F.array_join(F.array_intersect("reasons", reasons_with("review")), ";")
    clean = (rows.where(keep)
             .select("station_id", "station_name", "lat", "lon", "bucket",
                     (F.round(F.col("v") * 5) / 5).alias("value_mm"),
                     F.when(F.col("severity") == "review", review).alias("review"),
                     F.year("start").alias("year")))

    (clean.repartition("year")
          .sortWithinPartitions("station_id", "bucket")
          .write.mode("overwrite")
          .partitionBy("year")
          .parquet(f"{root}/clean"))
    print(f"wrote {root}/clean")

    # rows_kept_other_year: kept rows whose interval starts in another year than their file (review in the
    # audit as outside_file_year); they land in that other year's partition.
    summary = (rows.withColumn("kept", keep)
               .groupBy("file_year")
               .agg(F.count(F.lit(1)).alias("rows_read"),
                    F.sum((~F.col("kept")).cast("int")).alias("rows_dropped"),
                    F.sum(F.col("kept").cast("int")).alias("rows_kept"),
                    F.sum((F.col("kept") & (F.col("severity") == "review")).cast("int")).alias("rows_kept_review"),
                    F.sum((F.col("kept") & (F.year("start") != F.col("file_year"))).cast("int"))
                     .alias("rows_kept_other_year"))
               .orderBy("file_year"))
    summary.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{root}/clean_summary")
    summary.show(truncate=False)
    print(f"wrote {root}/clean_summary")


if __name__ == "__main__":
    main()
