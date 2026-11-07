#!/usr/bin/env bash
# Shell-only regression: no Conda environment, Python training or SLURM needed.
set -euo pipefail

repo_root="$(cd "${1:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd -P)"
temp_parent="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
test_root="$(mktemp -d "$temp_parent/labcompass-prior-test.XXXXXX")"
# Check the resolved absolute cleanup target before installing the trap.
test_root="$(cd "$test_root" && pwd -P)"
case "$test_root" in
    "$temp_parent"/labcompass-prior-test.*) ;;
    *) echo "Unsafe temporary directory: $test_root" >&2; exit 1 ;;
esac
trap 'rm -rf -- "$test_root"' EXIT

# Exported functions intercept the real launchers' external effects. In
# particular, source() prevents reading the caller's actual ~/.bashrc.
source() { printf 'source\n' >> "$LAUNCH_LOG"; }
conda() { printf 'conda:%s\n' "$*" >> "$LAUNCH_LOG"; }
python() {
    printf 'python\n' >> "$LAUNCH_LOG"
    printf 'arg:%s\n' "$@" >> "$LAUNCH_LOG"
}
export -f source conda python

passed=0
failed=0
run_case() {
    local launcher="$1" expected="$2" label="$3"
    shift 3
    local log="$test_root/trace" status=0
    : > "$log"
    LAUNCH_LOG="$log" SLURM_JOB_ID= SLURMD_NODENAME=test-node \
        bash --noprofile --norc "$launcher" "$@" \
        > "$test_root/stdout" 2> "$test_root/stderr" || status=$?
    if [ "$expected" = reject ]; then
        if [ "$status" -ne 2 ] || [ -s "$log" ] ||
            ! grep -Fq 'annotation=loop3' "$test_root/stderr"; then
            printf 'FAIL %s: expected exit 2 before source/conda/python, got %s\n' "$label" "$status" >&2
            return 1
        fi
    else
        local script expected_log="$test_root/expected"
        case "$launcher" in
            */train_prior.sbatch) script=train_flow_matching_prior.py ;;
            */train_flow_map.sbatch) script=distill_flow_map_prior.py ;;
            *) echo 'Unknown launcher' >&2; return 1 ;;
        esac
        {
            printf 'source\nconda:activate labcompass\npython\n'
            printf 'arg:%s\n' "$repo_root/scripts/prior/train/$script"
            if [ "$#" -gt 0 ]; then printf 'arg:%s\n' "$@"; fi
            printf 'arg:annotation=loop3\n'
        } > "$expected_log"
        if [ "$status" -ne 0 ] || [ -s "$test_root/stderr" ] ||
            ! cmp -s "$expected_log" "$log"; then
            printf 'FAIL %s: expected unchanged argument forwarding and exit 0, got %s\n' "$label" "$status" >&2
            return 1
        fi
    fi
    printf 'PASS %s\n' "$label"
}

for name in train_prior train_flow_map; do
    launcher="$repo_root/sbatch/sbatch_launchers/prior/$name.sbatch"
    for scenario in default matching matching-add matching-force unrelated \
        conflict conflict-add conflict-force empty mixed delete delete-value; do
        args=()
        expected=allow
        case "$scenario" in
            matching) args=(annotation=loop3) ;;
            matching-add) args=(+annotation=loop3) ;;
            matching-force) args=(++annotation=loop3) ;;
            unrelated) args=(paths=hpc_full 'paths.forward_model_checkpoint_path=/tmp/model with spaces.pkl' model.annotation=loop2) ;;
            conflict) expected=reject; args=(annotation=loop2) ;;
            conflict-add) expected=reject; args=(+annotation=loop2) ;;
            conflict-force) expected=reject; args=(++annotation=loop2) ;;
            empty) expected=reject; args=(annotation=) ;;
            mixed) expected=reject; args=(annotation=loop2 annotation=loop3) ;;
            delete) expected=reject; args=('~annotation') ;;
            delete-value) expected=reject; args=('~annotation=loop3') ;;
        esac
        if run_case "$launcher" "$expected" "$name/$scenario" "${args[@]}"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done
done
printf '%s passed; %s failed\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
