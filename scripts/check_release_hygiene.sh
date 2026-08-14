#!/usr/bin/env bash
# Check the candidate release tree for tracked local-only paths and common
# machine-specific or credential-like values. This intentionally checks the
# current tree, not the private development repository's historical commits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

failed=0
release_tree_mode="${SATNAV_RELEASE_TREE:-0}"
if [[ "${release_tree_mode}" != "0" && "${release_tree_mode}" != "1" ]]; then
    echo "[ERROR] SATNAV_RELEASE_TREE must be 0 or 1." >&2
    exit 2
fi

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

# Normal development checkouts are allowed to contain ignored overlays.  A
# sanitized release checkout can opt into a stricter existence check.  Only
# report the count: never read or print a user's ignored local files.
ignored_local_count=0
while IFS= read -r -d '' _local_file; do
    ((ignored_local_count += 1))
done < <(
    git ls-files -z --others --ignored --exclude-standard -- \
        '.local/**' \
        '**/.local/**' \
        '.env' \
        '**/.env' \
        'configs/local_*.yaml' \
        '*.local.yaml' \
        '**/*.local.yaml' \
        | sort -zu
)
if [[ "${release_tree_mode}" == "1" && "${ignored_local_count}" -ne 0 ]]; then
    echo "[ERROR] Sanitized release tree contains ${ignored_local_count} ignored local-only file(s)." >&2
    failed=1
fi
unset _local_file

candidate_files=()
absolute_path_files=()
while IFS= read -r -d '' file; do
    if [[ "/${file}/" == *"/rebuttal/"* ]]; then
        echo "[ERROR] Candidate release tree contains a forbidden comparison checkout." >&2
        failed=1
        continue
    fi
    if [[ "${file}" != "scripts/check_release_hygiene.sh" ]]; then
        if [[ -L "${file}" ]]; then
            target="$(readlink -f -- "${file}" 2>/dev/null || true)"
            case "${target}" in
                "${REPO_ROOT}"/*) ;;
                *)
                    echo "[ERROR] Candidate symlink points outside repository: ${file}" >&2
                    failed=1
                    continue
                    ;;
            esac
        fi
        candidate_files+=("${file}")
        absolute_path_files+=("${file}")
    fi
done < <(git ls-files -z --cached --others --exclude-standard)

credential_patterns=(
    '(^|[^0-9])(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})([^0-9]|$)'
    'qyapi\.weixin\.qq\.com/[^[:space:]]*key='
    '(gh[pousr]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})'
    '(github_pat_[A-Za-z0-9_]{20,}|sk-(proj-)?[A-Za-z0-9_-]{20,})'
    '(hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})'
    '-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'
    '(^|[^[:alnum:]_])[Bb]earer[[:space:]]+[A-Za-z0-9._~+/-]{12,}'
    'github\.com/yourusername'
)

report_matches() {
    local pattern="$1"
    shift
    grep -IHnE -- "${pattern}" "$@" 2>/dev/null \
        | sed -E 's/^([^:]+:[0-9]+):.*/\1/' \
        | sort -u || true
}

if (( ${#candidate_files[@]} > 0 )); then
    for pattern in "${credential_patterns[@]}"; do
        matches="$(report_matches "${pattern}" "${candidate_files[@]}")"
        if [[ -n "${matches}" ]]; then
            echo "[ERROR] Credential/private-host pattern matched:" >&2
            printf '%s\n' "${matches}" >&2
            failed=1
        fi
    done

    stale_pattern='ver_260202|ver_260211|ver_260317|satnav-refactor|SWIFTVLN_SATNAV_DATA_ROOT'
    matches="$(report_matches "${stale_pattern}" "${candidate_files[@]}")"
    if [[ -n "${matches}" ]]; then
        echo "[ERROR] Removed migration identifier remains:" >&2
        printf '%s\n' "${matches}" >&2
        failed=1
    fi
fi

if (( ${#absolute_path_files[@]} > 0 )); then
    absolute_patterns=(
        '(^|[^[:alnum:]_])(/mnt/|/root/|/Users/|/home/[^/[:space:]]+/|/data[0-9]+/|/srv/)'
        '(^|[^[:alnum:]_])[A-Za-z]:\\[^[:space:]]+'
    )
    for pattern in "${absolute_patterns[@]}"; do
        matches="$(report_matches "${pattern}" "${absolute_path_files[@]}")"
        if [[ -n "${matches}" ]]; then
            echo "[ERROR] Machine-specific absolute path matched:" >&2
            printf '%s\n' "${matches}" >&2
            failed=1
        fi
    done
fi

if (( failed != 0 )); then
    exit 1
fi

echo "[OK] Candidate release tree contains no tracked local-only paths or known local-value patterns."
