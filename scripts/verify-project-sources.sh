#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for relative_path in \
    "app/src/main/java/com/yy26git/depthanythingv2qnn/runtime/OnnxSessionFactory.kt" \
    "app/src/main/java/com/yy26git/depthanythingv2qnn/logging/Logger.kt"; do
    if [[ ! -f "${ROOT}/${relative_path}" ]]; then
        echo "FAIL missing required source: ${relative_path}" >&2
        exit 1
    fi
done
echo "Required application sources are present."
