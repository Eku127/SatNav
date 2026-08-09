#!/usr/bin/env bash
# Optional machine-local environment loader shared by SatNav entrypoints.
#
# Public entrypoints remain usable without local files. Machine paths and
# credentials can live in ignored .local/env.sh files instead of tracked
# scripts or YAML configs.

if [[ -z "${SATNAV_ROOT:-}" ]]; then
    _satnav_env_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SATNAV_ROOT="$(cd "${_satnav_env_lib_dir}/../.." && pwd)"
fi
export SATNAV_ROOT

satnav_load_local_env() {
    local component_dir="${1:-}"
    local shared_env_file="${SATNAV_LOCAL_ENV_FILE:-${SATNAV_ROOT}/.local/env.sh}"
    local component_env_file=""

    if [[ "${shared_env_file}" != /* ]]; then
        shared_env_file="${SATNAV_ROOT}/${shared_env_file}"
    fi
    if [[
        -r "${shared_env_file}"
        && "${SATNAV_LOADED_SHARED_ENV_FILE:-}" != "${shared_env_file}"
    ]]; then
        # shellcheck disable=SC1090
        source "${shared_env_file}"
        export SATNAV_LOADED_SHARED_ENV_FILE="${shared_env_file}"
    fi

    if [[ -n "${component_dir}" ]]; then
        component_env_file="${component_dir}/.local/env.sh"
        if [[
            -r "${component_env_file}"
            && "${component_env_file}" != "${shared_env_file}"
            && "${SATNAV_LOADED_COMPONENT_ENV_FILE:-}" != "${component_env_file}"
        ]]; then
            # shellcheck disable=SC1090
            source "${component_env_file}"
            export SATNAV_LOADED_COMPONENT_ENV_FILE="${component_env_file}"
        fi
    fi
}

unset _satnav_env_lib_dir
