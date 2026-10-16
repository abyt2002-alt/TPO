#!/bin/bash
set -euo pipefail
cd "${SOURCE_DIR:?}"
SHA="${GIT_SHA:?}"
[[ "${SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo "git_sha must be a full commit SHA"; exit 1; }
[[ "$(git rev-parse HEAD)" == "${SHA}" ]] || { echo "checkout does not match git_sha"; exit 1; }
PRODUCTION=c1551d15845e391b9b5726dbf866e1fd11caf04d
if [[ "${APPROVE_BRANCH:-}" == "APPROVE" ]]; then
  echo "Non-production revision ${SHA} was explicitly approved"
  exit 0
fi
git rev-parse --verify --quiet origin/Demolocal >/dev/null
if git merge-base --is-ancestor "${PRODUCTION}" "${SHA}" \
  && git merge-base --is-ancestor "${SHA}" origin/Demolocal; then
  echo "TPO revision ${SHA} is on Demolocal and does not predate production ${PRODUCTION}"
  exit 0
fi
echo "Refusing ${SHA}."
echo "Production is branch Demolocal at ${PRODUCTION}."
echo "origin/main has diverged and must not be deployed. Type APPROVE only after that difference is accepted."
exit 1
