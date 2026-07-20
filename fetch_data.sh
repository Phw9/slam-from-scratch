#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/image"
release_base="https://github.com/Phw9/slam-from-scratch/releases/download/kitti00-data"
force=0

if [[ "${1:-}" == "--force" ]]; then
    force=1
fi

fetch_archive() {
    local image_dir="$1"
    local archive_name="$2"
    local data_url="$3"
    local archive="$data_dir/$archive_name"

    if [[ "$force" -eq 0 && -d "$image_dir" &&
          -n "$(ls -A "$image_dir" 2>/dev/null)" ]]; then
        return 0
    fi

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
    tar --force-local -xzf "$archive" -C "$data_dir" 2>/dev/null ||
        tar -xzf "$archive" -C "$data_dir"
    rm -f "$archive"
    echo "data=ready image_dir=$image_dir files=$(ls "$image_dir" | wc -l)"
}

fetch_archive "$data_dir/image_0" "kitti00_image0.tar.gz" \
    "${MVO_DATA_URL:-$release_base/kitti00_image0.tar.gz}"
fetch_archive "$data_dir/image_1" "kitti00_image1.tar.gz" \
    "${MVO_STEREO_DATA_URL:-$release_base/kitti00_image1.tar.gz}"
