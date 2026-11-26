#!/usr/bin/env bash
# Run a command with the NVIDIA GPU hidden, rendering in software (Mesa llvmpipe).
args=(--dev-bind / / --tmpfs /dev/dri)
for d in /dev/nvidia*; do [ -c "$d" ] && args+=(--bind /dev/null "$d"); done
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4
unset GTK_PATH SNAP_COMMON
exec bwrap "${args[@]}" "$@"
