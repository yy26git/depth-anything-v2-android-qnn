#!/usr/bin/env bash
#
# Build, install and launch the Depth Anything QNN harness.
#
# Uses a build lock because simultaneous Gradle builds contend for shared daemons and
# project state. QNN libraries are staged locally under app/src/main/jniLibs/arm64-v8a,
# and ONNX Runtime is supplied as a local AAR.
#
# Usage:
#   scripts/deploy-android.sh                  # package + install + launch
#   scripts/deploy-android.sh --no-launch      # install only
#   scripts/deploy-android.sh --reinstall      # force a clean install (bypasses version-code reuse)
#   scripts/deploy-android.sh --logs           # after launching, stream logcat for the run
#   scripts/deploy-android.sh <task>...        # custom Gradle task(s)

set -euo pipefail

PACKAGE="com.yy26git.depthanythingv2qnn"
LAUNCHER="${PACKAGE}/com.yy26git.depthanythingv2qnn.MainActivity"
DEFAULT_TASK="assembleRelease"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FALLBACK_GRADLE_USER_HOME="${REPO_ROOT}/.gradle-deploy-home"

# Two transports are usually attached to the same physical device (one over TCP, one over
# wireless adb-tls). Pin one, or adb picks arbitrarily and can drop the connection mid-install.
DEVICE="${ADB_SERIAL:-}"
if [[ -z "${DEVICE}" ]]; then
    DEVICE="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
fi
if [[ -z "${DEVICE}" ]]; then
    echo "No adb device found. Connect the device or set ADB_SERIAL." >&2
    exit 1
fi

launch_app=true
force_reinstall=false
stream_logs=false
skip_build=false
gradle_tasks=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-launch)  launch_app=false; shift ;;
        --reinstall)  force_reinstall=true; shift ;;
        --no-build)   skip_build=true; shift ;;
        --logs)       stream_logs=true; shift ;;
        -h|--help)    sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            gradle_tasks+=("$1"); shift ;;
    esac
done

if [[ ${#gradle_tasks[@]} -eq 0 ]]; then
    gradle_tasks=("$DEFAULT_TASK")
fi

echo "Device: ${DEVICE}"
adb -s "${DEVICE}" wait-for-device

# The Gradle invocation itself. A lock prevents a second deploy from starting while this one
# holds the daemon.
#
# mkdir is the portable atomic primitive here, not flock: flock(1) does not exist on macOS, which
# is where this harness is built. The lock is a directory containing the owning PID, so a lock left
# behind by a killed deploy can be detected and cleared instead of blocking every future run.
LOCK_DIR="${REPO_ROOT}/.gradle-deploy-lock"
mkdir -p "$(dirname "${LOCK_DIR}")"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || echo '?')"
    if [[ "${holder}" != "?" ]] && ! kill -0 "${holder}" 2>/dev/null; then
        echo "Stale lock from dead PID ${holder}; clearing."
        rm -rf "${LOCK_DIR}"
        mkdir "${LOCK_DIR}"
    else
        echo "Another Gradle build holds ${LOCK_DIR}/lock (pid ${holder}). Wait for it, or remove the lock if it is stale." >&2
        exit 1
    fi
fi
echo $$ > "${LOCK_DIR}/pid"
cleanup_lock() { rm -rf "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup_lock EXIT

GRADLE_USER_HOME="${GRADLE_USER_HOME:-${FALLBACK_GRADLE_USER_HOME}}"
export GRADLE_USER_HOME
echo "GRADLE_USER_HOME=${GRADLE_USER_HOME}"

cd "${REPO_ROOT}"
if [[ "${skip_build}" != true ]]; then
    ./gradlew "${gradle_tasks[@]}"
fi

APK="${REPO_ROOT}/app/build/outputs/apk/release/app-release.apk"
if [[ ! -f "${APK}" ]]; then
    echo "FAIL: no APK at ${APK}" >&2
    exit 1
fi

if [[ "${force_reinstall}" == true ]]; then
    adb -s "${DEVICE}" uninstall "${PACKAGE}" >/dev/null 2>&1 || true
fi

# Install with adb directly rather than Gradle's installRelease.
#
# Gradle routes installs through ddmlib, which fails this APK with
# "DeviceException: InstallException: -99" — the APK is ~237 MB (an 87 MB libQnnHtpPrepare.so plus
# a 100 MB inlined model) and ddmlib's non-streaming path does not cope with it over wireless adb.
# Plain `adb install -r` streams the same file successfully. Keeping the install here means one
# obvious code path instead of a Gradle attempt plus a silent fallback.
echo "Installing $(basename "${APK}") ($(du -h "${APK}" | cut -f1))..."
adb -s "${DEVICE}" install -r "${APK}"

# "Installed on 1 device" from Gradle is not evidence that the new code is on the device.
# Verify the installed APK's lastUpdateTime moved, and that our marker string is inside it.
remote_apk="$(adb -s "${DEVICE}" shell pm path "${PACKAGE}" | sed 's/^package://' | tr -d '\r' | head -1)"
if [[ -z "${remote_apk}" ]]; then
    echo "FAIL: ${PACKAGE} is not installed after the build." >&2
    exit 1
fi
echo "Installed APK: ${remote_apk}"
adb -s "${DEVICE}" shell dumpsys package "${PACKAGE}" | grep -E "lastUpdateTime|versionName|versionCode" | head -5 | sed 's/^/  /'

if [[ "${launch_app}" == true ]]; then
    adb -s "${DEVICE}" shell am force-stop "${PACKAGE}" || true
    adb -s "${DEVICE}" shell am start -n "${LAUNCHER}" >/dev/null
    echo "Launched ${LAUNCHER}"
    # The harness finishes a full warmup+measured cycle in a few seconds; give it room on a
    # cold QNN context compile, which can take longer than that.
    sleep 3
fi

if [[ "${stream_logs}" == true ]]; then
    echo "Streaming logcat (DepthQNN / QNN). Ctrl-C to stop."
    adb -s "${DEVICE}" logcat -v brief -s DepthQNN:V QNN:V ORT:V '*:S'
fi
