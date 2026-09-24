#!/usr/bin/env bash
# Run a PySpark script from scripts/ on a temporary Dataproc cluster.
# Dataproc creates the cluster, runs the job and deletes the cluster, even if the job fails.
#
# Usage:
#   scripts/run_script.sh <script> [-- <script args>]
#
# Examples:
#   scripts/run_script.sh audit_dataset                     # all years
#   scripts/run_script.sh audit_dataset -- --years 2017     # one year, as a cheap test
#
# Settings (override with environment variables):
#   BUCKET_NAME   required, read from .env in the repo root (e.g. BUCKET_NAME=gs://my-bucket)
#   REGION        asia-southeast1
#   ZONE          asia-southeast1-a
#   MACHINE_TYPE  e2-highmem-8    (single node: 8 vCPU, 64 GB; the project quota is 12 vCPUs)
#   DISK_GB       100
#
# The script waits until the job finishes. Ctrl-C stops the waiting only: the job keeps running and
# the cluster is still deleted at the end. Follow it with:
#   gcloud dataproc operations list --region=$REGION
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load KEY=value lines from the repo's .env (read as text, never executed).
# A variable already set in the shell wins, so `BUCKET_NAME=... scripts/run_script.sh` still overrides it.
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value%$'\r'}"
    value="${value%\"}"; value="${value#\"}"
    [[ -n "${!key:-}" ]] || export "$key=$value"
  done < "$ENV_FILE"
fi

BUCKET_NAME="${BUCKET_NAME:?Set BUCKET_NAME in .env (e.g. BUCKET_NAME=gs://my-bucket)}"
REGION="${REGION:-asia-southeast1}"
ZONE="${ZONE:-asia-southeast1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-highmem-8}"
DISK_GB="${DISK_GB:-100}"
IMAGE_VERSION="2.2-debian12"

if [[ $# -lt 1 ]]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

JOB="${1%.py}"
shift
[[ "${1:-}" == "--" ]] && shift
[[ "$BUCKET_NAME" == gs://* ]] || BUCKET_NAME="gs://$BUCKET_NAME"

LOCAL_FILE="$SCRIPT_DIR/$JOB.py"
REMOTE_FILE="$BUCKET_NAME/code/$JOB.py"
if [[ ! -f "$LOCAL_FILE" ]]; then
  echo "No such script: $LOCAL_FILE" >&2
  exit 1
fi

# Cluster names: lowercase letters, digits and hyphens; unique per run so two runs never collide.
CLUSTER="rain-${JOB//_/-}-$(date +%m%d-%H%M%S)"
CLUSTER="$(echo "$CLUSTER" | tr '[:upper:]' '[:lower:]' | cut -c1-50)"

# The other scripts/*.py files are shipped as pyFiles, so jobs can import them (utils, audit_dataset, ...).
HELPERS=()
for f in "$SCRIPT_DIR"/*.py; do
  [[ "$f" == "$LOCAL_FILE" ]] || HELPERS+=("$f")
done
PYFILES_YAML=""
if [[ ${#HELPERS[@]} -gt 0 ]]; then
  PYFILES_YAML="      pythonFileUris:"$'\n'
  for f in "${HELPERS[@]}"; do
    PYFILES_YAML+="      - $BUCKET_NAME/code/$(basename "$f")"$'\n'
  done
fi

# Job arguments: --bucket first, then whatever was passed after "--".
ARGS=("--bucket" "$BUCKET_NAME" "$@")
ARGS_YAML=""
for a in "${ARGS[@]}"; do
  ARGS_YAML+="      - \"${a//\"/\\\"}\""$'\n'
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
WORKFLOW_FILE="$TMP_DIR/workflow.yaml"

cat > "$WORKFLOW_FILE" <<EOF
jobs:
  - stepId: ${JOB//_/-}
    pysparkJob:
      mainPythonFileUri: $REMOTE_FILE
${PYFILES_YAML}      args:
${ARGS_YAML}placement:
  managedCluster:
    clusterName: $CLUSTER
    config:
      gceClusterConfig:
        zoneUri: $ZONE
      masterConfig:
        numInstances: 1
        machineTypeUri: $MACHINE_TYPE
        diskConfig:
          bootDiskType: pd-balanced
          bootDiskSizeGb: $DISK_GB
      softwareConfig:
        imageVersion: $IMAGE_VERSION
        properties:
          "dataproc:dataproc.allow.zero.workers": "true"
          "spark:spark.sql.session.timeZone": "Asia/Singapore"
EOF

echo "== Uploading $LOCAL_FILE and ${#HELPERS[@]} helper module(s) -> $BUCKET_NAME/code/"
gcloud storage cp "$LOCAL_FILE" ${HELPERS[@]+"${HELPERS[@]}"} "$BUCKET_NAME/code/"

echo "== Workflow"
cat "$WORKFLOW_FILE"

echo "== Running on $CLUSTER ($MACHINE_TYPE, single node, $REGION). The cluster is deleted when the job ends."
START=$(date +%s)
gcloud dataproc workflow-templates instantiate-from-file \
  --file="$WORKFLOW_FILE" --region="$REGION"
echo "== Done in $(( ($(date +%s) - START) / 60 )) min. Outputs: $BUCKET_NAME"
