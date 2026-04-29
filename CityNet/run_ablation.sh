#!/bin/bash
# ============================================================
# run_ablation.sh
# Evaluate HRW vs RW across all supported graph types.
#
# Usage:
#   bash run_ablation.sh            # all experiments
#   bash run_ablation.sh barbell    # only experiments matching "barbell"
# ============================================================

set -euo pipefail

RUNNER="${RUNNER:-uv run python}"
mkdir -p ablation_logs

run() {
    local TAG=$1; shift
    echo ""
    echo "[$(date +%H:%M:%S)] ── $TAG"
    eval "$RUNNER ablation.py $*" 2>&1 | tee "ablation_logs/${TAG}.log"
    echo "[$(date +%H:%M:%S)] ── done: $TAG"
}

FILTER="${1:-}"

matches_filter() {
    local tag="$1"
    if [[ -z "$FILTER" ]]; then
        return 0
    fi
    [[ "$tag" == *"$FILTER"* ]]
}

maybe_run() {
    local tag="$1"; shift
    if matches_filter "$tag"; then
        run "$tag" "$@"
    fi
}

# ── Bottleneck-heavy (expect HRW to win clearly) ─────────────
uv run python hrw_analysis.py --graph barbell --R 4 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph sbm --R 3 --L 10 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph lollipop --R 4 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph grid --R 4 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph regular --R 3 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph er --R 3 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph caveman --R 6 --L 8 --M 3 --seed 42 --merge_terminal
uv run python hrw_analysis.py --graph ws --R 3 --L 10 --M 3 --seed 42 --merge_terminal


# ── Bottleneck-heavy (expect HRW to win clearly) ─────────────
# uv run python ablation.py --graph barbell --R 4 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph sbm --R 3 --L 10 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph lollipop --R 4 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph grid --R 4 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph regular --R 3 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph er --R 3 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph caveman --R 6 --L 8 --M 3 --seed 42 --merge_terminal
# uv run python ablation.py --graph ws --R 3 --L 10 --M 3 --seed 42 --merge_terminal


echo ""
echo "========================================================"
echo "All done. Logs: ablation_logs/"
echo "========================================================"