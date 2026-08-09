#!/usr/bin/env bash
# Check the candidate release tree for tracked local-only paths and common
# machine-specific or credential-like values. This intentionally checks the
# current tree, not the private development repository's historical commits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

failed=0

tracked_local_paths="$({
    git ls-files \
        '.codex/**' \
        '.agents/**' \
        '**/.local/**' \
        '.env' \
        '**/.env' \
        'configs/local_*.yaml' \
        '*.local.yaml' \
        '**/*.local.yaml' \
        'reports/**'
    git ls-files AGENTS.md
} | sort -u)"

if [[ -n "${tracked_local_paths}" ]]; then
    echo "[ERROR] Local-only paths are tracked:" >&2
    printf '%s\n' "${tracked_local_paths}" >&2
    failed=1
fi

candidate_files=()
while IFS= read -r -d '' file; do
    if [[ "${file}" != "scripts/check_release_hygiene.sh" ]]; then
        candidate_files+=("${file}")
    fi
done < <(git ls-files -z --cached --others --exclude-standard)

patterns=(
    '/mnt/data[0-9]*/'
    '/home/[^/[:space:]]+/(workspace|datasets?|miniconda|miniforge)/'
    '(^|[^0-9])(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})([^0-9]|$)'
    'qyapi\.weixin\.qq\.com/[^[:space:]]*key='
    '(gh[pousr]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,})'
    'github\.com/yourusername'
)

if (( ${#candidate_files[@]} > 0 )); then
    for pattern in "${patterns[@]}"; do
        matches="$(grep -IlE -- "${pattern}" "${candidate_files[@]}" || true)"
        if [[ -n "${matches}" ]]; then
            echo "[ERROR] Release-hygiene pattern matched these files:" >&2
            printf '%s\n' "${matches}" >&2
            failed=1
        fi
    done
fi

if (( failed != 0 )); then
    exit 1
fi

echo "[OK] Candidate release tree contains no tracked local-only paths or known local-value patterns."
