#!/usr/bin/env bash
set -Eeuo pipefail

# Publishes the current repository state to a separate personal GitHub remote.
# It deliberately leaves the upstream 'origin' untouched.
#
# Usage:
#   bash scripts/publish_github.sh git@github.com:USER/catsbench_mm.git
#   bash scripts/publish_github.sh https://github.com/USER/catsbench_mm.git catsbench-mm

REPOSITORY_URL="${1:-${GITHUB_REPOSITORY_URL:-}}"
TARGET_BRANCH="${2:-${GITHUB_BRANCH:-catsbench-mm}}"
REMOTE_NAME="${GITHUB_REMOTE_NAME:-personal}"
COMMIT_MESSAGE="${GITHUB_COMMIT_MESSAGE:-Prepare catsbench_mm for Colab experiments}"

if [[ -z "${REPOSITORY_URL}" ]]; then
  echo "Pass the URL of your empty GitHub repository as the first argument." >&2
  exit 2
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this script from inside the catsbench repository." >&2
  exit 1
}
cd "${PROJECT_ROOT}"

if [[ -z "$(git config user.name)" || -z "$(git config user.email)" ]]; then
  echo "Git identity is not configured. Configure it before publishing:" >&2
  echo '  git config user.name "Your Name"' >&2
  echo '  git config user.email "you@example.com"' >&2
  exit 1
fi

echo "Current branch: $(git branch --show-current)"
echo "Target: ${REMOTE_NAME}/${TARGET_BRANCH}"
echo "Files that are currently changed or untracked:"
git status --short
echo
read -r -p "Commit all changes shown above and push them to your repository? [y/N] " answer
case "${answer}" in
  y|Y|yes|YES) ;;
  *) echo "Cancelled; Git index and remotes were not changed."; exit 0 ;;
esac

if git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  configured_url="$(git remote get-url "${REMOTE_NAME}")"
  if [[ "${configured_url}" != "${REPOSITORY_URL}" ]]; then
    echo "Remote '${REMOTE_NAME}' already points to: ${configured_url}" >&2
    echo "Refusing to replace it automatically." >&2
    exit 1
  fi
else
  git remote add "${REMOTE_NAME}" "${REPOSITORY_URL}"
fi

git add --all
if ! git diff --cached --quiet; then
  git commit -m "${COMMIT_MESSAGE}"
else
  echo "No new changes to commit; publishing the current HEAD."
fi

git push --set-upstream "${REMOTE_NAME}" "HEAD:refs/heads/${TARGET_BRANCH}"
echo "Published to ${REPOSITORY_URL}, branch ${TARGET_BRANCH}."

