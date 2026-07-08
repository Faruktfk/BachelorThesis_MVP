#!/usr/bin/env bash

# ============================================================
# Run full Bachelor thesis ML-debugging experiment
#
# Default:
#   ./run0_experiments.sh
#
# Custom:
#   ./run0_experiments.sh itr_2 0 29
#
# Arguments:
#   $1 = iteration name / result folder name, default: itr_2
#   $2 = first seed, default: 0
#   $3 = last seed, default: 29
#
# Outputs:
#   results/<iteration>/log.md
#   results/<iteration>/experiments.csv
#   results/<iteration>/experiments.jsonl
#   results/<iteration>/run_manifest.md
#   results/<iteration>/failed_runs.tsv
# ============================================================

set -uo pipefail

# -----------------------------
# User-configurable parameters
# -----------------------------
ITERATION_NAME="${1:-itr_2}"
SEED_START="${2:-0}"
SEED_END="${3:-29}"

PYTHON_BIN="${PYTHON_BIN:-python}"

RESULT_DIR="results/${ITERATION_NAME}"
LOG_FILE="${RESULT_DIR}/log.md"
CSV_FILE="${RESULT_DIR}/experiments.csv"
JSONL_FILE="${RESULT_DIR}/experiments.jsonl"
MANIFEST_FILE="${RESULT_DIR}/run_manifest.md"
FAILED_RUNS_FILE="${RESULT_DIR}/failed_runs.tsv"

RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"

# Make Python behavior a bit more deterministic.
export PYTHONHASHSEED=0

# -----------------------------
# Experiment matrix
# -----------------------------
CASE_TITLES=(
  "Base experiment / no fault"
  "Label Noise Fault, random"
  "Label Noise Fault, hard"
  "Data Leakage Fault, direct"
  "Data Leakage Fault, indirect"
  "Spurious Correlation Fault, broken"
  "Spurious Correlation Fault, inverted"
)

CASE_HEADINGS=(
  "### 0.0 Base experiment / no fault"
  "### 1.1 Label Noise Fault, random"
  "### 1.2 Label Noise Fault, hard"
  "### 2.1 Data Leakage Fault, direct"
  "### 2.2 Data Leakage Fault, indirect"
  "### 3.1 Spurious Correlation Fault, broken"
  "### 3.2 Spurious Correlation Fault, inverted"
)

CASE_FAULTS=(
  "none"
  "label_noise"
  "label_noise"
  "data_leakage"
  "data_leakage"
  "spurious_correlation"
  "spurious_correlation"
)

CASE_MODES=(
  "0"
  "0"
  "1"
  "0"
  "1"
  "0"
  "1"
)

# -----------------------------
# Helper functions
# -----------------------------
print_console() {
  printf "%s\n" "$1"
}

append_log() {
  printf "%s\n" "$1" >> "$LOG_FILE"
}

seconds_to_hms() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

check_project_files() {
  local missing=0

  for file in main.py faults.py baseline_debugging.py xai_debugging.py evaluation_metrics.py; do
    if [[ ! -f "$file" ]]; then
      print_console "❌ Missing required file: $file"
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    print_console "Aborting because required project files are missing."
    exit 1
  fi
}

initialize_outputs() {
  mkdir -p "$RESULT_DIR"

  # Always start a clean run for this iteration.
  : > "$LOG_FILE"
  : > "$FAILED_RUNS_FILE"
  rm -f "$CSV_FILE" "$JSONL_FILE"

  printf "seed\tcase_title\tfault\tmode\texit_code\n" >> "$FAILED_RUNS_FILE"

  cat > "$MANIFEST_FILE" <<EOF
# Experiment Run Manifest

- Iteration: ${ITERATION_NAME}
- Run ID: ${RUN_ID}
- Started at: ${RUN_STARTED_AT}
- Seed range: ${SEED_START} to ${SEED_END}
- Python command: ${PYTHON_BIN}
- Main output log: ${LOG_FILE}
- CSV output: ${CSV_FILE}
- JSONL output: ${JSONL_FILE}

## Experiment Cases

| Case | Fault | Mode |
|---|---|---|
EOF

  local i
  for i in "${!CASE_TITLES[@]}"; do
    printf "| %s | %s | %s |\n" "${CASE_TITLES[$i]}" "${CASE_FAULTS[$i]}" "${CASE_MODES[$i]}" >> "$MANIFEST_FILE"
  done

  cat >> "$LOG_FILE" <<EOF
# Full Experiment Log

- Iteration: ${ITERATION_NAME}
- Run ID: ${RUN_ID}
- Started at: ${RUN_STARTED_AT}
- Seed range: ${SEED_START} to ${SEED_END}
- CSV output: ${CSV_FILE}
- JSONL output: ${JSONL_FILE}

EOF
}

run_one_case() {
  local seed="$1"
  local case_index="$2"

  local title="${CASE_TITLES[$case_index]}"
  local heading="${CASE_HEADINGS[$case_index]}"
  local fault="${CASE_FAULTS[$case_index]}"
  local mode="${CASE_MODES[$case_index]}"

  print_console "▶ Seed ${seed} | ${title}"

  append_log ""
  append_log "${heading}"
  append_log ""
  append_log "\`\`\`text"

  local start_seconds
  start_seconds="$(date +%s)"

  "$PYTHON_BIN" -u main.py \
    --fault "$fault" \
    --mode "$mode" \
    --seed "$seed" \
    --output-csv "$CSV_FILE" \
    --output-jsonl "$JSONL_FILE" \
    >> "$LOG_FILE" 2>&1

  local exit_code=$?

  local end_seconds
  end_seconds="$(date +%s)"
  local duration=$((end_seconds - start_seconds))

  append_log "\`\`\`"
  append_log ""
  append_log "- Exit code: ${exit_code}"
  append_log "- Runtime: $(seconds_to_hms "$duration")"
  append_log ""
  append_log "---"
  append_log ""

  if [[ "$exit_code" -eq 0 ]]; then
    print_console "  done ✅ ($(seconds_to_hms "$duration"))"
  else
    print_console "  failed ❌ exit_code=${exit_code} ($(seconds_to_hms "$duration"))"
    printf "%s\t%s\t%s\t%s\t%s\n" "$seed" "$title" "$fault" "$mode" "$exit_code" >> "$FAILED_RUNS_FILE"
  fi

  return "$exit_code"
}

# -----------------------------
# Main script
# -----------------------------
check_project_files
initialize_outputs

total_cases=${#CASE_TITLES[@]}
total_seeds=$((SEED_END - SEED_START + 1))
total_runs=$((total_cases * total_seeds))
current_run=0
failed_count=0

overall_start_seconds="$(date +%s)"

print_console "============================================================"
print_console "Starting experiment run: ${ITERATION_NAME}"
print_console "Seeds: ${SEED_START} to ${SEED_END}"
print_console "Total runs: ${total_runs}"
print_console "Results: ${RESULT_DIR}"
print_console "============================================================"

for seed in $(seq "$SEED_START" "$SEED_END"); do
  print_console ""
  print_console "==================== SEED ${seed} ===================="

  append_log ""
  append_log ""
  append_log "=========== SEED: ${seed} ============="
  append_log ""
  append_log "## SEED: ${seed}"
  append_log ""

  for case_index in "${!CASE_TITLES[@]}"; do
    current_run=$((current_run + 1))
    print_console "Progress: ${current_run}/${total_runs}"

    if ! run_one_case "$seed" "$case_index"; then
      failed_count=$((failed_count + 1))
    fi
  done
done

overall_end_seconds="$(date +%s)"
overall_duration=$((overall_end_seconds - overall_start_seconds))
RUN_FINISHED_AT="$(date '+%Y-%m-%d %H:%M:%S')"

cat >> "$MANIFEST_FILE" <<EOF

## Run Summary

- Finished at: ${RUN_FINISHED_AT}
- Total planned runs: ${total_runs}
- Failed runs: ${failed_count}
- Total runtime: $(seconds_to_hms "$overall_duration")

## Output Files

- Log: \`${LOG_FILE}\`
- CSV: \`${CSV_FILE}\`
- JSONL: \`${JSONL_FILE}\`
- Failed runs: \`${FAILED_RUNS_FILE}\`
EOF

append_log ""
append_log "## Final Run Summary"
append_log ""
append_log "- Finished at: ${RUN_FINISHED_AT}"
append_log "- Total planned runs: ${total_runs}"
append_log "- Failed runs: ${failed_count}"
append_log "- Total runtime: $(seconds_to_hms "$overall_duration")"
append_log "- CSV output: ${CSV_FILE}"
append_log "- JSONL output: ${JSONL_FILE}"
append_log ""

print_console ""
print_console "============================================================"
print_console "Experiment finished."
print_console "Total runtime: $(seconds_to_hms "$overall_duration")"
print_console "Failed runs: ${failed_count}"
print_console "Log: ${LOG_FILE}"
print_console "CSV: ${CSV_FILE}"
print_console "JSONL: ${JSONL_FILE}"
print_console "============================================================"

if [[ "$failed_count" -gt 0 ]]; then
  print_console "Some runs failed. Check: ${FAILED_RUNS_FILE}"
  exit 1
fi


python analyze_results.py --input results/${ITERATION_NAME}/experiments.csv --outdir results/${ITERATION_NAME}/analysis

exit 0