#!/usr/bin/env bash
set -Eeuo pipefail

# Additive file exchange between the current clone and mounted Google Drive.
# Usage:
#   bash scripts/sync_google_drive.sh push              # push logs
#   bash scripts/sync_google_drive.sh push logs data    # push selected paths
#   bash scripts/sync_google_drive.sh pull logs         # restore selected paths

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DRIVE_PROJECT_DIR="${CATS_DRIVE_DIR:-/content/drive/MyDrive/catsbench_mm}"
ACTION="${1:-push}"

if [[ "${ACTION}" != "push" && "${ACTION}" != "pull" ]]; then
  echo "Usage: $0 {push|pull} [relative_path ...]" >&2
  exit 2
fi
shift || true

if [[ ! -d "$(dirname -- "${DRIVE_PROJECT_DIR}")" ]]; then
  echo "Google Drive is not mounted: $(dirname -- "${DRIVE_PROJECT_DIR}")" >&2
  echo "Mount Drive in Colab first, then rerun this command." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in this runtime." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  set -- logs
fi

mkdir -p "${DRIVE_PROJECT_DIR}"

for relative_path in "$@"; do
  case "${relative_path}" in
    ""|/*|../*|*/../*|*/..)
      echo "Only safe project-relative paths are allowed: ${relative_path}" >&2
      exit 2
      ;;
  esac

  if [[ "${ACTION}" == "push" ]]; then
    source_path="${PROJECT_ROOT}/${relative_path}"
    target_path="${DRIVE_PROJECT_DIR}/${relative_path}"
  else
    source_path="${DRIVE_PROJECT_DIR}/${relative_path}"
    target_path="${PROJECT_ROOT}/${relative_path}"
  fi

  if [[ ! -e "${source_path}" ]]; then
    echo "Skipping missing path: ${source_path}" >&2
    continue
  fi

  mkdir -p "$(dirname -- "${target_path}")"
  if [[ -d "${source_path}" ]]; then
    mkdir -p "${target_path}"
    rsync -a --info=stats2 \
      --exclude='.git/' --exclude='.venv/' \
      "${source_path}/" "${target_path}/"
  else
    rsync -a --info=stats2 "${source_path}" "${target_path}"
  fi
done

echo "${ACTION} completed: ${DRIVE_PROJECT_DIR}"

