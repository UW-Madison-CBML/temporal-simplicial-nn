#!/usr/bin/env bash
# Download and unpack the preprocessed simplicial-complex extracts.
#
# The raw SEED-VII recordings are NOT redistributed here: they require their own
# access agreement from the dataset authors. What is hosted is the derived
# windowed extract (per-window simplicial complexes and features) that the
# training code consumes. To regenerate it yourself from raw SEED-VII instead,
# see model/seed_to_sccn_window.py.
#
#   ./scripts/download_data.sh
#   DATA_URL=https://osf.io/xxxxx/download ./scripts/download_data.sh
#
# ~2.5 GB download, ~2.5 GB unpacked.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
ARCHIVE_NAME="seedvii_simplicial_extracts.tar.gz"

# ---------------------------------------------------------------------------
# TODO: set this to the OSF download link for the archive before release, e.g.
#   DATA_URL="https://osf.io/<file-id>/download"
# Optionally set EXPECTED_SHA256 to the checksum published alongside it.
# ---------------------------------------------------------------------------
DATA_URL="${DATA_URL:-}"
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

EXTRACTS=(all_band all_band_eye delta theta alpha beta gamma)

have_all () {
  for e in "${EXTRACTS[@]}"; do [[ -d "${DATA_ROOT}/${e}/clique" ]] || return 1; done
  return 0
}

if have_all; then
  echo "All extracts already present in ${DATA_ROOT}; nothing to do."
  exit 0
fi

if [[ -z "$DATA_URL" ]]; then
  cat >&2 <<'MSG'
error: DATA_URL is not set.

  Edit scripts/download_data.sh and set DATA_URL to the archive link, or run:
      DATA_URL=https://osf.io/<file-id>/download ./scripts/download_data.sh

  If you already downloaded the archive by hand, unpack it with:
      tar -xzf seedvii_simplicial_extracts.tar.gz -C data/
MSG
  exit 1
fi

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
mkdir -p "$DATA_ROOT"
archive="${DATA_ROOT}/${ARCHIVE_NAME}"

echo "Downloading ~2.5 GB ..."
# -L follows redirects (OSF uses them); -C - resumes an interrupted download
curl -fL -C - --retry 3 --retry-delay 5 -o "$archive" "$DATA_URL"

if [[ -n "$EXPECTED_SHA256" ]]; then
  echo "Verifying checksum ..."
  actual="$(sha256sum "$archive" | cut -d' ' -f1)"
  if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    echo "error: checksum mismatch" >&2
    echo "  expected $EXPECTED_SHA256" >&2
    echo "  actual   $actual" >&2
    echo "  the download is incomplete or corrupt; delete $archive and retry" >&2
    exit 1
  fi
  echo "  ok"
fi

echo "Unpacking into ${DATA_ROOT} ..."
tar -xzf "$archive" -C "$DATA_ROOT"
rm -f "$archive"

echo
echo "Extracts present:"
for d in "${DATA_ROOT}"/*/; do
  [[ -d "${d}clique" ]] && printf "  %-14s %s\n" "$(basename "$d")" "$(du -sh "$d" | cut -f1)"
done
have_all || echo "warning: some extracts are missing; the archive may be incomplete" >&2
