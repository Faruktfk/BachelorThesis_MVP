#!/usr/bin/env bash

# ============================================================
# Bundle source code + experiment outputs into one review file
#
# Default:
#   ./make_code_log.sh
#
# Custom:
#   ./make_code_log.sh itr_7
#
# Output:
#   results/<iteration>/code_log_<iteration>.txt
# ============================================================

set -uo pipefail

ITERATION_NAME="${1:-itr_7}"
RESULT_DIR="results/${ITERATION_NAME}"

LOG_FILE="${RESULT_DIR}/log.md"
CSV_FILE="${RESULT_DIR}/experiments.csv"
JSONL_FILE="${RESULT_DIR}/experiments.jsonl"
MANIFEST_FILE="${RESULT_DIR}/run_manifest.md"
FAILED_RUNS_FILE="${RESULT_DIR}/failed_runs.tsv"

OUTPUT_FILE="${RESULT_DIR}/code_log_${ITERATION_NAME}.txt"

INCLUDE_JSONL="${INCLUDE_JSONL:-0}"

SOURCE_FILES=(
  "main.py"
  "faults.py"
  "baseline_debugging.py"
  "xai_debugging.py"
  "evaluation_metrics.py"
)

print_console() {
  printf "%s\n" "$1"
}

write_section_header() {
  local title="$1"

  {
    printf "\n\n"
    printf "##########################\n"
    printf "# %s\n" "$title"
    printf "##########################\n"
    printf "\n"
  } >> "$OUTPUT_FILE"
}

append_file_or_warning() {
  local title="$1"
  local file_path="$2"

  write_section_header "$title"

  if [[ -f "$file_path" ]]; then
    cat "$file_path" >> "$OUTPUT_FILE"
  else
    printf "WARNING: File not found: %s\n" "$file_path" >> "$OUTPUT_FILE"
  fi
}

append_file_metadata() {
  write_section_header "File Metadata"

  {
    printf "Iteration: %s\n" "$ITERATION_NAME"
    printf "Created at: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "Result dir: %s\n" "$RESULT_DIR"
    printf "\n"

    printf "Source files:\n"
    for file in "${SOURCE_FILES[@]}"; do
      if [[ -f "$file" ]]; then
        printf -- "- %s | lines=%s | bytes=%s\n" "$file" "$(wc -l < "$file")" "$(wc -c < "$file")"
      else
        printf -- "- %s | MISSING\n" "$file"
      fi
    done

    printf "\nResult files:\n"
    for file in "$MANIFEST_FILE" "$LOG_FILE" "$CSV_FILE" "$JSONL_FILE" "$FAILED_RUNS_FILE"; do
      if [[ -f "$file" ]]; then
        printf -- "- %s | lines=%s | bytes=%s\n" "$file" "$(wc -l < "$file")" "$(wc -c < "$file")"
      else
        printf -- "- %s | MISSING\n" "$file"
      fi
    done
  } >> "$OUTPUT_FILE"
}

mkdir -p "$RESULT_DIR"
: > "$OUTPUT_FILE"

append_file_metadata

for file in "${SOURCE_FILES[@]}"; do
  append_file_or_warning "$file" "$file"
done

append_file_or_warning "run_manifest.md" "$MANIFEST_FILE"
append_file_or_warning "failed_runs.tsv" "$FAILED_RUNS_FILE"
append_file_or_warning "experiments.csv" "$CSV_FILE"
append_file_or_warning "log.md" "$LOG_FILE"

if [[ "$INCLUDE_JSONL" == "1" ]]; then
  append_file_or_warning "experiments.jsonl" "$JSONL_FILE"
else
  write_section_header "experiments.jsonl"
  {
    printf "JSONL was not included by default because it can become very large.\n"
    printf "To include it, run:\n"
    printf "INCLUDE_JSONL=1 ./make_code_log.sh %s\n" "$ITERATION_NAME"
  } >> "$OUTPUT_FILE"
fi

print_console "done ✅"
print_console "Created bundle: ${OUTPUT_FILE}"

start .