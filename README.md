# DSA5208-rainfall

Spark analysis of NEA's [Historical Rainfall across Singapore](https://data.gov.sg/) dataset: 5-minute rainfall at
every rain gauge, 2017–2024 (about 45–50 million rows, 9 GB of CSV).

- **Task 1** describes rainfall patterns within and across years, and their relationship with ENSO.
- **Task 2** predicts P(rainfall at a station in the next 30 minutes > τ mm) with one gradient-boosted tree model that
  takes the threshold τ as an input, calibrated with isotonic regression. Results are reported for τ = 0, 0.2, 1, 5
  and 10 mm.

**Try the model:** open [`prototype.ipynb`](prototype.ipynb) in
[Colab](https://colab.research.google.com/github/Yanaput/DSA5208-rainfall/blob/main/prototype.ipynb) and run it from
top to bottom. It needs no cloud account.

## Where each file runs

| File | Runs on | What it does |
|---|---|---|
| `download_dataset.ipynb` | Colab (or local) | Step 0: downloads each year's CSV from the data.gov.sg API into `gs://<bucket>/raw/` |
| `scripts/run_script.sh` | Local terminal → **GCP** | Runs one of the scripts below as a job on a temporary Dataproc cluster, which is deleted afterwards |
| `scripts/audit_dataset.py` | **GCP** (Dataproc) | Step 1: validates the raw data; writes `audit/` |
| `scripts/clean_dataset.py` | **GCP** (Dataproc) | Step 2: dedupes, snaps and drops invalid rows; writes `clean/` (Parquet by year) |
| `scripts/build_features.py` | **GCP** (Dataproc) | Step 3a: features and 30-minute labels per station and 5-minute time; writes `features/`, `stations/` |
| `scripts/train.py` | **GCP** (Dataproc) | Step 3b: `--mode tune` (rolling-origin CV, writes `tune/`) and `--mode final` (GBT + calibrator, writes `models/`) |
| `scripts/predict.py` | **GCP** (Dataproc) | Step 3c: `--evaluate` tests on 2024 and writes `results/`; also answers a single query |
| `scripts/utils.py` | (imported) | Shared constants used by the jobs |
| `task1_describe.ipynb` | Colab | Task 1: aggregates `clean/` with PySpark in local mode; makes the Task 1 figures and CSVs |
| `task1_audit_plot.ipynb` | Colab or local | Stations-reporting figure from `audit/`, and the annual table from the Task 1 CSVs |
| `scripts/plot_model_diagram.py` | Local | Model-construction diagrams (`figures/4*.png`) |
| `scripts/plot_test_auc.py` | Local | ROC and PR curves on the 2024 test set (`figures/5_test_roc_pr.png`) from `results/predictions` |
| `prototype.ipynb` | Colab or local | Runs the saved model from `prototype/` on 2024 data: one query, the threshold curve, an island map, a day |

`figures/` holds the figures used in the report. `prototype/` holds a copy of the trained model (`models/`) and the
2024 features, so the prototype runs without the bucket.

## Setup

**Local Python** (Python 3.10+):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install openjdk@17     # only for PySpark in local mode (macOS; use your package manager elsewhere)
```

**GCP** (for the pipeline): install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), run
`gcloud auth login`, and create a `.env` file in the repository root (it is gitignored):

```bash
BUCKET_NAME=gs://<your-bucket>
```

The notebooks' saved outputs show the original bucket, `gs://dsa5208-yanaput-rain`. It is private, so to rerun the
pipeline, download the data into your own bucket with `download_dataset.ipynb`. The prototype doesn't need a bucket.

`run_script.sh` creates a cluster of 1 master and 2 workers (12 vCPUs) in `asia-southeast1`. `REGION`, `ZONE`,
`MASTER_TYPE`, `WORKER_TYPE`, `NUM_WORKERS` and `DISK_GB` can be overridden as environment variables.

**Colab:** add the secret `DSA5208_GS_BUCKET` (e.g. `gs://my-bucket`), read by `download_dataset.ipynb` and
`task1_describe.ipynb`, and `DATAGOVSG` (a data.gov.sg API key), read by `download_dataset.ipynb`.
`prototype.ipynb` needs no setup.

## Running the pipeline

Each step reads the previous step's output from the bucket and writes under `gs://<bucket>/<step>/`.

```bash
# 0. Ingest: run download_dataset.ipynb (Colab), which fills gs://<bucket>/raw/

# 1-2. Validate and clean (append `-- --years 2017` to any job for a cheap one-year test)
scripts/run_script.sh audit_dataset
scripts/run_script.sh clean_dataset

# 3. Features, tuning, final model, test on 2024
scripts/run_script.sh build_features
scripts/run_script.sh train -- --mode tune --sample-days 0.2 --max-depth 5 7 --max-iter 100 200
scripts/run_script.sh train -- --mode final --max-depth 5 --max-iter 200 --drop day_of_year
scripts/run_script.sh predict -- --evaluate

# One prediction from the saved model (station, Singapore time, thresholds in mm)
scripts/run_script.sh predict -- --station S24 --time "2024-04-15 16:00" --threshold 1 5
```

Task 1 runs separately in `task1_describe.ipynb` on Colab once `clean/` exists.

To redraw the test-set curves locally, copy the predictions first:

```bash
gsutil -m cp -r gs://<bucket>/results/predictions data/results/
python scripts/plot_test_auc.py
```

## Predicting years after 2024

The trained model is not tied to a year: its inputs are recent rain at the station, its neighbours and across the
island, time of day, month, position and the threshold, and its climatology comes from the saved 2017–2022 counts.
What limits it is the input data:

- **Features exist only for 2017–2024**, and `predict.py` (and the prototype) look a station's row up in `features/`.
- **`build_features.py` needs a whole year**: a station is used only if it reported in at least 10% of that year's
  5-minute slots, and neighbour lists are built per year from those stations.
- **Each row needs the next 30 minutes** to be complete, because the same row carries the label used for evaluation.

So the pipeline predicts past times, not the future.

- **A later year, after the fact** (like the 2024 test): download the year with `download_dataset.ipynb`, add it to
  `YEARS` in `scripts/utils.py`, and rerun `audit_dataset`, `clean_dataset` and `build_features` for that year and
  the one before it, since look-back windows cross 1 January. The model is used as trained.
- **Live forecasts** would need a separate feature step that reads NEA's real-time readings, uses a fixed station and
  neighbour list, and builds the row at time t without the next 30 minutes.

The further from the training years, the more the network and rainfall can drift: new gauges have no climatology
and fall back to the island-wide rate. Recalibrating on the most recent full year, as was done with 2023, keeps the
probabilities reliable.

## Data rules

These are easy to get wrong:

- Timestamps are Singapore time (`+08:00`). Every Spark session sets `spark.sql.session.timeZone=Asia/Singapore`;
  on a UTC session, hour and month features shift by 8 hours.
- Timestamps are rounded **up** to the 5-minute mark *before* deduplicating: in April 2017 the label format changed
  from `hh:x4:59` to `hh:x5:00`, so the same reading appears under two labels one second apart.
- Duplicates on `(station_id, bucket)` keep the latest `reading_update_timestamp`.
- Values are snapped to the 0.2 mm grid (`round(v * 5) / 5`) before any threshold comparison.
- Stations are keyed on `station_id` only: names are not unique (`S24` and `S24B`), and coordinates can move.
- A missing reading is a missing row, not a null. Gaps are never treated as dry: Task 1 uses complete days only, and
  a prediction row is dropped if any of the next six readings is missing.
- Split: train on 2017–2022 (folds validate on 2020, 2021 and 2022), calibrate and choose cut-offs on 2023, and test
  on 2024 only. Climatology features come from the training years only.
