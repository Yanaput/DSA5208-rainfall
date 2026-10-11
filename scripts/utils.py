import os

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
BUCKET_NAME = os.environ.get("BUCKET_NAME")

BUCKET_S = 300                          # 5 minutes
MIN_COVERAGE = 0.10                     # stations below this share of a year's buckets are not used that year
