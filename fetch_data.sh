#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/image"
image_dir="$data_dir/image_0"
data_url="${MVO_DATA_URL:-https://github.com/Phw9/mvo/releases/download/kitti00-data/kitti00_image0.tar.gz}"
force=0

if [[ "${1:-}" == "--force" ]]; then
    force=1
fi

if [[ "$force" -eq 0 && -d "$image_dir" && -n "$(ls -A "$image_dir" 2>/dev/null)" ]]; then
    exit 0
fi

archive="$data_dir/kitti00_image0.tar.gz"
mkdir -p "$data_dir"
echo "data=downloading url=$data_url"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$archive" "$data_url"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive" "$data_url"
else
    echo "error: curl or wget is required to download the image data" >&2
    exit 1
fi
echo "data=extracting archive=$archive"
tar --force-local -xzf "$archive" -C "$data_dir" 2>/dev/null || tar -xzf "$archive" -C "$data_dir"
rm -f "$archive"
echo "data=ready image_dir=$image_dir files=$(ls "$image_dir" | wc -l)"
