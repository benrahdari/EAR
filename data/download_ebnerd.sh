#!/usr/bin/env bash
# Download and unpack EB-NeRD into data/. usage: bash data/download_ebnerd.sh [small|large|demo] ...
# Default: small. "large" is a 3.2 GB zip. Files come from the dataset's public S3 bucket.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://ebnerd-dataset.s3.eu-west-1.amazonaws.com
for size in "${@:-small}"; do
  curl -fL -o "ebnerd_${size}.zip" "${BASE}/ebnerd_${size}.zip"
  mkdir -p "ebnerd_${size}" && unzip -oq "ebnerd_${size}.zip" -x "__MACOSX/*" -d "ebnerd_${size}" && rm "ebnerd_${size}.zip"
done
