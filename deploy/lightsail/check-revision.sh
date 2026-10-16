#!/bin/bash
set -euo pipefail
cd "${SOURCE_DIR:?}"
SHA="${GIT_SHA:?}"
[[ "${SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo "git_sha must be a full commit SHA"; exit 1; }
[[ "$(git rev-parse HEAD)" == "${SHA}" ]] || { echo "checkout does not match git_sha"; exit 1; }
PRODUCTION=c1551d15845e391b9b5726dbf866e1fd11caf04d
if [[ "${SHA}" == "${PRODUCTION}" ]]; then
  echo "TPO revision matches production branch Demolocal"
  exit 0
fi
if [[ "${APPROVE_BRANCH:-}" == "APPROVE" ]]; then
  echo "Non-production revision ${SHA} was explicitly approved"
  exit 0
fi
echo "Refusing ${SHA}."
echo "Production is branch Demolocal at ${PRODUCTION}."
echo "origin/main at e3c2483c513b04ad5a1224783534aa917b59daf5 has diverged: 6 production commits are absent from main and 5 main commits are absent from production."
echo "Automatic main deployment stays disabled. Type APPROVE only after that difference is accepted."
exit 1
