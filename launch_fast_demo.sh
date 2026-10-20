#!/usr/bin/env bash
# Downloads only the desktop interface. Docker, models, and full code wait for Download.
set -Eeuo pipefail
export LEVIR_LAUNCHER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export LEVIR_LAUNCHER_FILE="$LEVIR_LAUNCHER_ROOT/$(basename -- "${BASH_SOURCE[0]}")"
RAW='https://raw.githubusercontent.com/QinzheYang/LEVIRDet-release/main'
VERSION=44.5.1
RUNTIME_SHA='5bcd217611d6843ececd6c9e9c1fcd1da3ab066c43d8b1a9e4b44689a1fba6f5'
RUNTIME_SIZE=122932200
RUNTIME_NAME='electron-v44.5.1-linux-x64.zip'
RUNTIME_URL="https://github.com/electron/electron/releases/download/v$VERSION/$RUNTIME_NAME"
MANUAL='Project: https://github.com/QinzheYang/LEVIRDet-release'
say(){ printf '[LEVIRDetNet] %s\n' "$*"; }
fail(){
    trap - ERR
    local help="$LEVIR_LAUNCHER_ROOT/LEVIRDetNet-launcher-help.txt"
    printf 'LEVIRDetNet launcher could not start.\n\n%s\n\n%s\n' "$*" "$MANUAL" > "$help" || true
    printf '%s\nInstructions: %s\n' "$*" "$help" >&2
    if command -v zenity >/dev/null 2>&1; then zenity --error --title='LEVIRDetNet launcher' --text="$*
Instructions: $help" 2>/dev/null || true
    elif command -v kdialog >/dev/null 2>&1; then kdialog --title='LEVIRDetNet launcher' --error "$*
Instructions: $help" 2>/dev/null || true; fi
    exit 1
}
trap 'fail "Launcher preparation failed at line $LINENO. Review the download and manual paths in the help file."' ERR
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This launcher requires a Linux x86-64 desktop. Use launch_fast_demo.cmd on Windows.'
[[ ${BASH_VERSINFO[0]} -ge 4 ]] || fail 'Bash 4 or newer is required.'
[[ $EUID -ne 0 ]] || fail 'Open the launcher as an ordinary desktop user, not root. It requests administrator authentication only for installation you approve.'
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] || fail 'No graphical desktop was found. Open this script in your Linux desktop session.'
REQUIRED=(fast_demo/assets.json fast_demo/launcher/package.json fast_demo/launcher/main.cjs fast_demo/launcher/preload.cjs fast_demo/launcher/controller.cjs fast_demo/launcher/transfers.cjs fast_demo/launcher/index.html fast_demo/launcher/renderer.js fast_demo/launcher/styles.css fast_demo/launcher/prerequisites.ps1 fast_demo/launcher/prerequisites.sh)
is_launcher(){
    local entry
    for entry in "${REQUIRED[@]}"; do [[ -f "$1/$entry" ]] || return 1; done
    # Check capability as text without executing the downloaded JavaScript.
    grep -Eq '"launcher_download_version"[[:space:]]*:[[:space:]]*([3-9]|[1-9][0-9]+)[[:space:]]*[,}]' "$1/fast_demo/assets.json" || return 1
    for entry in controller.cjs transfers.cjs; do
        grep -Eq '(^|[^[:alnum:]_])downloadWithMirrors([^[:alnum:]_]|$)' "$1/fast_demo/launcher/$entry" || return 1
    done
}
download(){
    local url="$1" destination="$2" resume="${3:-false}"; local -a options=()
    MANUAL+=$'\n\n'"Download: $url"$'\n'"Save as: $destination"
    command -v curl >/dev/null 2>&1 || fail 'curl is needed for the initial desktop download. Install curl using your distribution package manager and reopen this script.'
    mkdir -p -- "$(dirname -- "$destination")"
    [[ "$resume" != true || ! -s "$destination.part" ]] || options=(--continue-at -)
    say "Preparing the desktop interface: $(basename -- "$destination")"
    curl --location --fail --show-error --progress-bar --retry 3 --connect-timeout 30 --speed-limit 1024 --speed-time 120 "${options[@]}" --output "$destination.part" "$url" || return 1
}
APP_ROOT=''
for candidate in "$LEVIR_LAUNCHER_ROOT" "$LEVIR_LAUNCHER_ROOT/levirdetnet-release" "$LEVIR_LAUNCHER_ROOT/LEVIRDet-release-main" "$LEVIR_LAUNCHER_ROOT/.levir-launcher-v3"; do
    if is_launcher "$candidate"; then APP_ROOT="$candidate"; break; fi
done
if [[ -z "$APP_ROOT" ]]; then
    APP_ROOT="$LEVIR_LAUNCHER_ROOT/.levir-launcher-v3"
    for file in "${REQUIRED[@]}"; do
        destination="$APP_ROOT/$file"
        if [[ ! -f "$destination" ]]; then
            download "$RAW/$file" "$destination" || fail "Cannot download $file. Ensure the new launcher files are published on GitHub main."
            [[ -s "$destination.part" ]] || fail "Empty launcher download: $file"
            if [[ "$file" != *.html ]] && head -c 256 -- "$destination.part" | grep -aiEq '^\s*(<!doctype\s+html|<html)'; then fail "Unexpected web page instead of $file"; fi
            mv -- "$destination.part" "$destination"
        fi
    done
    for file in fast_demo/launcher/assets/background.png fast_demo/launcher/assets/logo.png fast_demo/launcher/assets/wordmark.png; do
        destination="$APP_ROOT/$file"
        if [[ ! -f "$destination" ]]; then
            if download "$RAW/$file" "$destination"; then mv -- "$destination.part" "$destination"; else say "Optional artwork unavailable: $file"; fi
        fi
    done
fi
is_launcher "$APP_ROOT" || fail "The launcher files do not contain the current Hugging Face download support. Publish the updated fast_demo files to GitHub main, then move this incomplete launcher cache aside and reopen this script: $APP_ROOT"
ELECTRON=''
for candidate in "$LEVIR_LAUNCHER_ROOT" "$LEVIR_LAUNCHER_ROOT/levirdetnet-release" "$LEVIR_LAUNCHER_ROOT/LEVIRDet-release-main" "$LEVIR_LAUNCHER_ROOT/.levir-launcher" "$APP_ROOT"; do
    directory="$candidate/.fast_demo/electron/linux-x64/$VERSION"
    if [[ -x "$directory/electron" && -f "$directory/.ready" && "$(cat -- "$directory/.ready")" == "$RUNTIME_SHA" ]]; then ELECTRON="$directory/electron"; break; fi
done
if [[ -z "$ELECTRON" ]]; then
    for program in unzip sha256sum stat; do command -v "$program" >/dev/null 2>&1 || fail "Missing desktop bootstrap tool: $program. Install it with your distribution package manager."; done
    CACHE="$APP_ROOT/.fast_demo_cache"; mkdir -p -- "$CACHE"
    archive="$CACHE/$RUNTIME_NAME"
    MANUAL+=$'\n\n'"Desktop runtime: $RUNTIME_URL"$'\n'"Save as: $archive"$'\n'"SHA-256: $RUNTIME_SHA"
    if [[ ! -f "$archive" ]]; then
        if [[ ! -f "$archive.part" || "$(stat -c %s -- "$archive.part")" != "$RUNTIME_SIZE" ]]; then download "$RUNTIME_URL" "$archive" true || fail 'Could not download the desktop runtime. Partial data is kept for the next attempt.'; fi
        sum="$(sha256sum -- "$archive.part")"; sum="${sum%% *}"
        [[ "$(stat -c %s -- "$archive.part")" == "$RUNTIME_SIZE" && "$sum" == "$RUNTIME_SHA" ]] || fail "Desktop runtime download failed verification: $archive.part. Move it aside and download the exact runtime manually."
        mv -- "$archive.part" "$archive"
    fi
    say 'Verifying the desktop runtime...'
    sum="$(sha256sum -- "$archive")"; sum="${sum%% *}"
    [[ "$(stat -c %s -- "$archive")" == "$RUNTIME_SIZE" && "$sum" == "$RUNTIME_SHA" ]] || fail "Desktop runtime ZIP failed verification: $archive"
    directory="$APP_ROOT/.fast_demo/electron/linux-x64/$VERSION"
    [[ ! -e "$directory" ]] || fail "Desktop runtime is incomplete. Move this directory aside before retrying: $directory"
    while IFS= read -r name; do [[ "$name" != /* && "$name" != *\\* && "$name" != *:* && "/$name/" != *'/../'* ]] || fail 'Unsafe desktop runtime ZIP entry.'; done < <(unzip -Z1 "$archive")
    if unzip -Z -l "$archive" | awk '$1 ~ /^l/ {found=1} END {exit !found}'; then fail 'The runtime ZIP contains unexpected symbolic links.'; fi
    mkdir -p -- "$directory"; unzip -q "$archive" -d "$directory"
    chmod u+x -- "$directory/electron"
    printf '%s' "$RUNTIME_SHA" > "$directory/.ready"
    ELECTRON="$directory/electron"
fi
say 'Opening LEVIRDetNet Launcher. Docker and model downloads start only after you choose Download.'
cd -- "$APP_ROOT"
exec "$ELECTRON" "$APP_ROOT/fast_demo/launcher"
