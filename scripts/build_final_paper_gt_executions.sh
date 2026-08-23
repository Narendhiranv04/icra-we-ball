#!/usr/bin/env bash
# Run every GT variant twice (timing-only and natural-speed recording), then
# package the evidence into one final-paper directory.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/naren/miniconda3/bin/python}"
FINAL_ROOT="${FINAL_ROOT:-${REPO_ROOT}/FINAL_PAPER_GT_EXECUTIONS}"
TILE_RESOLUTION="${TILE_RESOLUTION:-960x540}"
FPS="${FPS:-20}"
REPLACE_FINAL=0
REPLACE_VARIANT=0
SHOW_LIVE=0
SELECTED_ENVIRONMENT=""
SELECTED_VARIANT=""

usage() {
  printf 'Run GT variants twice, record five views, and package final evidence.\n'
  printf '\nUsage:\n'
  printf '  %s [--replace-final]\n' "$0"
  printf '  %s --environment {kitchen|living_room|workshop} --variant NAME [--replace-variant]\n' "$0"
  printf '\nOptions:\n'
  printf '  --show  Display the five-view mosaic during both timing and recorded passes.\n'
}

while (($#)); do
  case "$1" in
    --replace-final) REPLACE_FINAL=1; shift ;;
    --replace-variant) REPLACE_VARIANT=1; shift ;;
    --show) SHOW_LIVE=1; shift ;;
    --environment)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      SELECTED_ENVIRONMENT="$2"; shift 2 ;;
    --variant)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      SELECTED_VARIANT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -n "$SELECTED_ENVIRONMENT" || -n "$SELECTED_VARIANT" ]]; then
  [[ -n "$SELECTED_ENVIRONMENT" && -n "$SELECTED_VARIANT" ]] || {
    printf '%s\n' '--environment and --variant must be supplied together.' >&2
    exit 2
  }
  [[ "$REPLACE_FINAL" -eq 0 ]] || {
    printf '%s\n' '--replace-final cannot be combined with single-variant mode.' >&2
    exit 2
  }
else
  [[ "$REPLACE_VARIANT" -eq 0 ]] || {
    printf '%s\n' '--replace-variant requires single-variant mode.' >&2
    exit 2
  }
fi

cd "$REPO_ROOT"
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
command -v ffprobe >/dev/null || { printf 'ffprobe is required.\n' >&2; exit 1; }
if [[ "$SHOW_LIVE" -eq 1 ]]; then
  command -v ffplay >/dev/null || { printf 'ffplay is required for --show.\n' >&2; exit 1; }
  [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] || {
    printf '%s\n' '--show requires a graphical desktop session (DISPLAY or WAYLAND_DISPLAY).' >&2
    exit 1
  }
fi

if [[ -z "$SELECTED_VARIANT" && -e "$FINAL_ROOT" && "$REPLACE_FINAL" -ne 1 ]]; then
  printf 'Refusing to overwrite %s. Re-run with --replace-final.\n' "$FINAL_ROOT" >&2
  exit 1
fi
case "$FINAL_ROOT" in
  "$REPO_ROOT"/FINAL_PAPER_GT_EXECUTIONS) ;;
  *) printf 'FINAL_ROOT must resolve to %s/FINAL_PAPER_GT_EXECUTIONS\n' "$REPO_ROOT" >&2; exit 1 ;;
esac

RAW_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ra-final-paper-gt.XXXXXX")"
STAGING_ROOT="$(mktemp -d "${REPO_ROOT}/.final-paper-gt-package.XXXXXX")"
PACKAGE_ROOT="${STAGING_ROOT}/FINAL_PAPER_GT_EXECUTIONS"
cleanup() {
  find "$RAW_ROOT" -depth -delete 2>/dev/null || true
  find "$STAGING_ROOT" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT

RECORDED_ROOT="${RAW_ROOT}/recorded"
UNRECORDED_ROOT="${RAW_ROOT}/unrecorded"
TIMINGS_ROOT="${RAW_ROOT}/timings"
LOGS_ROOT="${RAW_ROOT}/logs"
mkdir -p "$RECORDED_ROOT" "$UNRECORDED_ROOT" "$TIMINGS_ROOT" "$LOGS_ROOT"

readarray -t KITCHEN_VARIANTS < <(
  "$PYTHON_BIN" -c 'from mujoco_scenes.run_kitchen_ground_truth_execution import discover_variant_names; print("\n".join(discover_variant_names()))'
)
readarray -t LIVING_VARIANTS < <(
  "$PYTHON_BIN" -c 'from mujoco_scenes.run_living_room_execution import EXPECTED_VARIANTS; print("\n".join(EXPECTED_VARIANTS))'
)
readarray -t WORKSHOP_VARIANTS < <(
  "$PYTHON_BIN" -c 'from mujoco_scenes.workshop_ground_truth_planner import load_variant_specs; print("\n".join(load_variant_specs()))'
)

[[ "${#KITCHEN_VARIANTS[@]}" -eq 16 ]] || {
  printf 'Expected 16 Kitchen variants, found %d.\n' "${#KITCHEN_VARIANTS[@]}" >&2
  exit 1
}
[[ "${#LIVING_VARIANTS[@]}" -eq 10 ]] || {
  printf 'Expected 10 Living Room variants, found %d.\n' "${#LIVING_VARIANTS[@]}" >&2
  exit 1
}
[[ "${#WORKSHOP_VARIANTS[@]}" -eq 10 ]] || {
  printf 'Expected 10 Workshop variants, found %d.\n' "${#WORKSHOP_VARIANTS[@]}" >&2
  exit 1
}

if [[ -n "$SELECTED_VARIANT" ]]; then
  case "$SELECTED_ENVIRONMENT" in
    kitchen) SELECTED_VARIANTS=("${KITCHEN_VARIANTS[@]}") ;;
    living_room) SELECTED_VARIANTS=("${LIVING_VARIANTS[@]}") ;;
    workshop) SELECTED_VARIANTS=("${WORKSHOP_VARIANTS[@]}") ;;
    *) printf 'Unknown environment: %s\n' "$SELECTED_ENVIRONMENT" >&2; exit 2 ;;
  esac
  variant_found=0
  for candidate in "${SELECTED_VARIANTS[@]}"; do
    [[ "$candidate" == "$SELECTED_VARIANT" ]] && variant_found=1
  done
  [[ "$variant_found" -eq 1 ]] || {
    printf 'Unknown %s variant: %s\n' "$SELECTED_ENVIRONMENT" "$SELECTED_VARIANT" >&2
    exit 2
  }
  if [[ -e "$FINAL_ROOT/$SELECTED_ENVIRONMENT/$SELECTED_VARIANT" && "$REPLACE_VARIANT" -ne 1 ]]; then
    printf 'Refusing to overwrite existing final variant: %s\n' \
      "$FINAL_ROOT/$SELECTED_ENVIRONMENT/$SELECTED_VARIANT" >&2
    printf '%s\n' 'Re-run with --replace-variant to replace only this variant.' >&2
    exit 1
  fi
fi

run_timed() {
  local environment="$1"
  local variant="$2"
  local mode="$3"
  shift 3
  mkdir -p "$TIMINGS_ROOT/$environment" "$LOGS_ROOT/$environment"
  printf '[%s] %s (%s)\n' "$environment" "$variant" "$mode"
  local started ended command_status timing_path
  timing_path="$TIMINGS_ROOT/$environment/${variant}_${mode}_seconds.txt"
  started="$("$PYTHON_BIN" -c 'import time; print(time.monotonic())')"
  set +e
  env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PYTHON_BIN" "$@" \
    >"$LOGS_ROOT/$environment/${variant}_${mode}.log" 2>&1
  command_status=$?
  set -e
  ended="$("$PYTHON_BIN" -c 'import time; print(time.monotonic())')"
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(f"{float(sys.argv[3]) - float(sys.argv[2]):.9f}\n", encoding="utf-8")' \
    "$timing_path" "$started" "$ended"
  if [[ "$command_status" -ne 0 ]]; then
    printf '\nERROR: %s %s failed during %s (exit %d).\n' \
      "$environment" "$variant" "$mode" "$command_status" >&2
    printf 'Execution log: %s\n\n' "$LOGS_ROOT/$environment/${variant}_${mode}.log" >&2
    sed -n '1,240p' "$LOGS_ROOT/$environment/${variant}_${mode}.log" >&2
  fi
  return "$command_status"
}

run_kitchen() {
  local variant="$1"
  local mode="$2"
  local output="$3"
  local record_args=()
  [[ "$mode" == "with_recording" ]] && record_args=(--record)
  [[ "$SHOW_LIVE" -eq 1 ]] && record_args+=(--show)
  run_timed kitchen "$variant" "$mode" \
    -m mujoco_scenes.run_kitchen_ground_truth_execution \
    --variant "$variant" --output-root "$output" \
    --speed 1.0 --strict-robot-execution --fps "$FPS" \
    --camera-resolution "$TILE_RESOLUTION" "${record_args[@]}"
}

run_living() {
  local variant="$1"
  local mode="$2"
  local output="$3"
  local record_args=()
  [[ "$mode" == "with_recording" ]] && record_args=(--record)
  [[ "$SHOW_LIVE" -eq 1 ]] && record_args+=(--show)
  run_timed living_room "$variant" "$mode" \
    -m mujoco_scenes.run_living_room_execution \
    --variant "$variant" --output-root "$output" \
    --fps "$FPS" --tile-resolution "$TILE_RESOLUTION" \
    "${record_args[@]}"
}

run_workshop() {
  local variant="$1"
  local mode="$2"
  local output="$3"
  local record_args=()
  [[ "$mode" == "with_recording" ]] && record_args=(--record)
  [[ "$SHOW_LIVE" -eq 1 ]] && record_args+=(--show)
  run_timed workshop "$variant" "$mode" \
    -m mujoco_scenes.run_workshop_ground_truth_execution \
    --variant "$variant" --output-root "$output" \
    --assignment-source oracle \
    --fps "$FPS" --resolution "$TILE_RESOLUTION" "${record_args[@]}"
}

if [[ -n "$SELECTED_VARIANT" ]]; then
  case "$SELECTED_ENVIRONMENT" in
    kitchen)
      run_kitchen "$SELECTED_VARIANT" without_recording "$UNRECORDED_ROOT/kitchen"
      run_kitchen "$SELECTED_VARIANT" with_recording "$RECORDED_ROOT/kitchen"
      ;;
    living_room)
      run_living "$SELECTED_VARIANT" without_recording "$UNRECORDED_ROOT/living_room"
      run_living "$SELECTED_VARIANT" with_recording "$RECORDED_ROOT/living_room"
      ;;
    workshop)
      run_workshop "$SELECTED_VARIANT" without_recording "$UNRECORDED_ROOT/workshop"
      run_workshop "$SELECTED_VARIANT" with_recording "$RECORDED_ROOT/workshop"
      ;;
  esac
else
  # First pass: physics execution without video recording. With --show, the
  # five-view mosaic is displayed live and the resulting wall time includes
  # live rendering, exactly as requested for interactive review.
  for variant in "${KITCHEN_VARIANTS[@]}"; do
    run_kitchen "$variant" without_recording "$UNRECORDED_ROOT/kitchen"
  done
  for variant in "${LIVING_VARIANTS[@]}"; do
    run_living "$variant" without_recording "$UNRECORDED_ROOT/living_room"
  done
  for variant in "${WORKSHOP_VARIANTS[@]}"; do
    run_workshop "$variant" without_recording "$UNRECORDED_ROOT/workshop"
  done

  # Second pass: the same GT execution with direct natural-time five-view capture.
  for variant in "${KITCHEN_VARIANTS[@]}"; do
    run_kitchen "$variant" with_recording "$RECORDED_ROOT/kitchen"
  done
  for variant in "${LIVING_VARIANTS[@]}"; do
    run_living "$variant" with_recording "$RECORDED_ROOT/living_room"
  done
  for variant in "${WORKSHOP_VARIANTS[@]}"; do
    run_workshop "$variant" with_recording "$RECORDED_ROOT/workshop"
  done
fi

if [[ -n "$SELECTED_VARIANT" ]]; then
  package_args=(
    --recorded-root "$RECORDED_ROOT"
    --unrecorded-root "$UNRECORDED_ROOT"
    --timings-root "$TIMINGS_ROOT"
    --output-root "$FINAL_ROOT"
    --environment "$SELECTED_ENVIRONMENT"
    --variant "$SELECTED_VARIANT"
    --append
  )
  [[ "$REPLACE_VARIANT" -eq 1 ]] && package_args+=(--replace-existing)
  "$PYTHON_BIN" -m mujoco_scenes.package_final_paper_gt "${package_args[@]}"
  printf '\nFinal paper variant created at: %s\n' \
    "$FINAL_ROOT/$SELECTED_ENVIRONMENT/$SELECTED_VARIANT"
  exit 0
fi

"$PYTHON_BIN" -m mujoco_scenes.package_final_paper_gt \
  --recorded-root "$RECORDED_ROOT" \
  --unrecorded-root "$UNRECORDED_ROOT" \
  --timings-root "$TIMINGS_ROOT" \
  --output-root "$PACKAGE_ROOT"

if [[ -e "$FINAL_ROOT" ]]; then
  find "$FINAL_ROOT" -depth -delete
fi
mv "$PACKAGE_ROOT" "$FINAL_ROOT"
printf '\nFinal paper evidence created at: %s\n' "$FINAL_ROOT"
