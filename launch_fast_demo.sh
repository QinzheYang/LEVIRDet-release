#!/usr/bin/env bash
# Download this file, put it in a writable folder, and open it with Bash.
# Linux x86-64 desktop, NVIDIA GPU, and GPU-enabled Docker are required.
set -Eeuo pipefail
LAUNCHER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY='https://github.com/QinzheYang/LEVIRDet-release'
ARCHIVE='https://codeload.github.com/QinzheYang/LEVIRDet-release/zip/refs/heads/main'
PROJECT='' CONTAINER='' MANUAL='' DOWNLOADED=''
say() { printf '[LEVIRDetNet] %s\n' "$*"; }
manual() { MANUAL+="$*"$'\n\n'; }
manual "Project: $REPOSITORY
Download the complete repository ZIP, extract it as levirdetnet-release beside this launcher, and run the launcher again."
manual 'Linux GPU container setup: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Install Docker Engine/Desktop, the NVIDIA driver, and NVIDIA Container Toolkit. Your ordinary desktop user must have permission to run Docker.'
cleanup() {
    if [[ -n "$CONTAINER" ]]; then
        say 'Stopping this demo container...'
        docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
        docker rm "$CONTAINER" >/dev/null 2>&1 || true
    fi
}
fail() {
    trap - ERR
    local reason="$*" help="$LAUNCHER_ROOT/LEVIRDetNet-setup-help.txt"
    printf '\nSetup stopped: %s\n\n%s' "$reason" "$MANUAL" >&2
    { printf 'LEVIRDetNet Fast Demo - setup help\n\n%s\n\n%s\nRun the launcher again after completing these steps. Existing project files are not overwritten.\n' "$reason" "$MANUAL" > "$help"; } 2>/dev/null || true
    if command -v zenity >/dev/null 2>&1; then
        zenity --warning --title='LEVIRDetNet setup' --text="$reason

Instructions: $help" 2>/dev/null || true
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --title 'LEVIRDetNet setup' --sorry "$reason

Instructions: $help" 2>/dev/null || true
    fi
    exit 1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
trap 'fail "An operation failed at launcher line $LINENO. Check the error above and the setup instructions below."' ERR

is_project() {
    local base="$1" entry
    for entry in demo/image_demo.py mmdet/apis/levir_image_demo.py configs/_base_/levirdetnet.py configs/levirdetnet/levirdetnet-30class.py configs/levirdetnet/levirdetnet-159class.py fast_demo/server.py fast_demo/assets.json fast_demo/desktop/main.cjs fast_demo/desktop/package.json fast_demo/desktop/preload.cjs fast_demo/static/index.html fast_demo/static/app.js fast_demo/static/style.css; do
        [[ -f "$base/$entry" ]] || return 1
    done
}
html_file() {
    LC_ALL=C head -c 1024 -- "$1" | LC_ALL=C grep -aiEq '^[[:space:]]*(<!doctype[[:space:]]+html|<html|<\?xml)'
}
urlencode() {
    local value="$1" encoded='' char hex i
    LC_ALL=C
    for ((i=0; i<${#value}; i++)); do
        char="${value:i:1}"
        case "$char" in [a-zA-Z0-9.~_-]) encoded+="$char";; *) printf -v hex '%%%02X' "'$char"; encoded+="$hex";; esac
    done
    printf '%s' "$encoded"
}
download_file() {
    local url="$1" destination="$2" drive_id="${3:-}" part="$2.part" cookies="$2.cookies" attempt html confirmation key value
    local -a extra=()
    if [[ -n "$drive_id" ]]; then url="https://drive.usercontent.google.com/download?id=$drive_id&export=download&authuser=0&confirm=t"; fi
    for ((attempt=0; attempt<3; attempt++)); do
        extra=()
        if [[ -s "$part" ]]; then
            if html_file "$part"; then rm -- "$part"; else extra=(--continue-at -); fi
        fi
        if ! curl --location --fail --show-error --retry 3 --retry-delay 2 --connect-timeout 30 --speed-limit 1024 --speed-time 120 --cookie "$cookies" --cookie-jar "$cookies" "${extra[@]}" --output "$part" "$url"; then
            fail "Download failed: $destination. Partial bytes remain in $part for the next attempt. Check network/proxy/Google Drive quota, or use the manual download instructions below."
        fi
        if ! html_file "$part"; then DOWNLOADED="$part"; return; fi
        [[ -n "$drive_id" ]] || fail 'GitHub returned an HTML page instead of the project ZIP.'
        [[ "$(stat -c %s -- "$part")" -le 2097152 ]] || fail 'The download returned an unexpected HTML page.'
        # Google Drive's large-file warning contains a GET form with hidden fields.
        # Accept only the known Google download endpoint, never arbitrary page URLs.
        grep -q 'drive.usercontent.google.com/download' "$part" || fail 'Google Drive returned a permission, quota, or sign-in page. Download the file manually using the link below.'
        confirmation=''
        while IFS=$'\t' read -r key value; do
            case "$key" in id|export|confirm|uuid|at)
                [[ -n "$confirmation" ]] && confirmation+='&'
                confirmation+="$(urlencode "$key")=$(urlencode "$value")";;
            esac
        done < <(awk '
            BEGIN { RS="<" }
            /^input[[:space:]]/ {
                tag=$0; name=""; value="";
                while (match(tag, /[[:alnum:]_-]+[[:space:]]*=[[:space:]]*("[^"]*"|\047[^\047]*\047)/)) {
                    a=substr(tag,RSTART,RLENGTH); tag=substr(tag,RSTART+RLENGTH);
                    k=a; sub(/[[:space:]]*=.*/,"",k); sub(/^[^=]*=[[:space:]]*/,"",a); a=substr(a,2,length(a)-2);
                    gsub(/&amp;/,"\&",a); gsub(/&quot;/,"\"",a); gsub(/&#39;/,"\047",a);
                    if (k=="name") name=a; if (k=="value") value=a;
                }
                if (name!="") print name "\t" value;
            }
        ' "$part")
        [[ "$confirmation" == *confirm=* ]] || fail 'Google Drive did not provide a usable large-file confirmation. Use the manual download link below.'
        url="https://drive.usercontent.google.com/download?$confirmation"
        rm -- "$part"
    done
    fail 'Google Drive confirmation did not produce a file. Use the manual download link below.'
}
safe_unzip() {
    local archive="$1" destination="$2" name
    while IFS= read -r name; do
        [[ "$name" != /* && "$name" != *\\* && "$name" != *:* && "/$name/" != *'/../'* ]] || fail 'The ZIP contains an unsafe path.'
    done < <(unzip -Z1 "$archive")
    if unzip -Z -l "$archive" | awk '$1 ~ /^l/ {found=1} END {exit !found}'; then fail 'The ZIP contains symbolic links; extract a trusted distribution manually instead.'; fi
    unzip -q "$archive" -d "$destination" || fail 'ZIP extraction failed. Check that the download is complete and disk space is available.'
}
ensure_project() {
    local candidate target="$LAUNCHER_ROOT/levirdetnet-release" extract
    for candidate in "$LAUNCHER_ROOT" "$LAUNCHER_ROOT/levirdetnet-release" "$LAUNCHER_ROOT/LEVIRDet-release-main"; do
        if is_project "$candidate"; then PROJECT="$candidate"; return; fi
    done
    [[ ! -e "$target" ]] || fail "The existing project is incomplete: $target. Add the missing Fast Demo/project files, or move this launcher to a fresh folder. Existing files will not be overwritten."
    say 'Downloading the complete project from GitHub...'
    download_file "$ARCHIVE" "$LAUNCHER_ROOT/LEVIRDet-release-main.zip"
    extract="$(mktemp -d "$LAUNCHER_ROOT/.fast-demo-code.XXXXXXXX")"
    safe_unzip "$DOWNLOADED" "$extract"
    local -a children=("$extract"/*)
    [[ ${#children[@]} -eq 1 ]] && is_project "${children[0]}" || fail "The public repository does not yet contain the complete Fast Demo. Download/update the project from $REPOSITORY and try again."
    mv -- "${children[0]}" "$target"
    PROJECT="$target"
}
# A small JSON reader keeps the launcher independent of Python, Node, and jq.
# Output is a flattened path/value TSV; release manifest keys/values use ASCII.
parse_manifest() {
    awk '
    function bad() { print "Invalid or unsupported asset manifest" > "/dev/stderr"; exit 2 }
    function ws() { while(substr(s,p,1) ~ /[ \t\r\n]/ && p<=length(s)) p++ }
    function str(   out,c,e) {
        if(substr(s,p++,1)!="\"") bad(); out="";
        while(p<=length(s)) {
            c=substr(s,p++,1); if(c=="\"") return out;
            if(c=="\\") { e=substr(s,p++,1); if(e=="\""||e=="\\"||e=="/") c=e; else if(e=="n") c=" "; else if(e=="r") c=" "; else if(e=="t") c=" "; else bad() }
            if(c ~ /[\t\r\n]/) bad(); out=out c;
        } bad()
    }
    function value(path,  c,key,i,begin,v) {
        ws(); c=substr(s,p,1);
        if(c=="{") { p++; ws(); if(substr(s,p,1)=="}") {p++; return}; while(1) {ws(); key=str(); ws(); if(substr(s,p++,1)!=":")bad(); value(path "/" key); ws(); c=substr(s,p++,1); if(c=="}")break; if(c!=",")bad()} }
        else if(c=="[") {p++; ws(); i=0; if(substr(s,p,1)=="]"){p++;return}; while(1){value(path "/" i++);ws();c=substr(s,p++,1);if(c=="]")break;if(c!=",")bad()}}
        else if(c=="\"") {v=str(); print path "\t" v}
        else {begin=p;while(substr(s,p,1) ~ /[[:alnum:].+_-]/ && p<=length(s))p++;v=substr(s,begin,p-begin);if(v!~/^(-?[0-9]+([.][0-9]+)?|true|false|null)$/)bad();print path "\t" v}
    }
    {s=s $0 "\n"}
    END {p=1;value("");ws();if(p<=length(s))bad()}
    ' "$PROJECT/fast_demo/assets.json"
}
manifest_value() {
    awk -F '\t' -v key="$1" '$1==key {print substr($0,index($0,"\t")+1); found++;} END {if(found!=1) exit 2}' "$CACHE/manifest.tsv"
}
asset_path() {
    local relative="$1"
    [[ -n "$relative" && "$relative" != /* && "$relative" != *\\* && "$relative" != *:* && "/$relative/" != *'/../'* ]] || fail 'The manifest contains an unsafe asset path.'
    printf '%s/%s' "$PROJECT" "$relative"
}
verify_file() {
    local id="$1" file="$2" size="$3" expected="$4" stamp hash record
    [[ -f "$file" ]] || return 1
    [[ -z "$size" || "$(stat -c %s -- "$file")" == "$size" ]] || return 1
    stamp="$(stat -c '%s|%y' -- "$file")"
    record="$file|$stamp|$expected"
    if [[ -f "$CACHE/$id.verified" && "$(cat -- "$CACHE/$id.verified")" == "$record" ]]; then return; fi
    say "Checking SHA-256: $(basename -- "$file") (large files take a few minutes on first use)..."
    hash="$(sha256sum -- "$file")"; hash="${hash%% *}"
    [[ "$hash" == "$expected" ]] || return 1
    printf '%s\n' "$record" > "$CACHE/$id.verified"
}
cache_moved_file() {
    local id="$1" file="$2" expected="$3"
    printf '%s|%s|%s\n' "$file" "$(stat -c '%s|%y' -- "$file")" "$expected" > "$CACHE/$id.verified"
}

printf '\n\033[40;97m  LEVIRDetNet Fast Demo  \033[0m\n\n'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This launcher requires a Linux x86-64 desktop with an NVIDIA GPU. Use launch_fast_demo.cmd on Windows.'
[[ ${BASH_VERSINFO[0]} -ge 4 ]] || fail 'Bash 4 or newer is required.'
[[ $EUID -ne 0 ]] || fail 'Run this desktop demo as an ordinary desktop user, not root. Configure Docker access for that user; the launcher does not disable the Electron sandbox.'
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] || fail 'No desktop display was found. Open this launcher in a local Linux desktop session; an SSH-only/headless server cannot show the demo window.'
for program in curl unzip awk sha256sum stat head grep mktemp od tr id docker; do
    command -v "$program" >/dev/null 2>&1 || fail "Missing prerequisite: $program. Install it using your Linux distribution package manager, then reopen the launcher. No Conda or Python installation is required."
done
docker info >/dev/null 2>&1 || fail 'Docker is not running or this user cannot access it. Start Docker and configure access for your ordinary desktop user.'
[[ "$(docker info --format '{{.OSType}}')" == linux ]] || fail 'Docker must use Linux containers.'
ensure_project
say "Project: $PROJECT"
CACHE="$PROJECT/.fast_demo_cache"
mkdir -p -- "$CACHE"
parse_manifest > "$CACHE/manifest.tsv"
[[ "$(manifest_value /schema_version)" == 1 ]] || fail 'Unsupported asset manifest. Download the latest complete project.'
IMAGE="$(manifest_value /docker_image)"
[[ "$IMAGE" == 'levir-train:cuda121-torch231' ]] || fail 'Unexpected Docker image in the asset manifest.'
declare -a IDS PATHS URLS DRIVE_IDS SIZES HASHES
declare -A SEEN
for i in 0 1 2; do
    IDS[i]="$(manifest_value "/assets/$i/id")"
    case "${IDS[i]}" in docker|weights30|weights159) ;; *) fail 'The manifest must provide docker, weights30, and weights159.';; esac
    [[ -z "${SEEN[${IDS[i]}]:-}" ]] || fail 'The asset manifest has duplicate IDs.'
    SEEN[${IDS[i]}]=1
    PATHS[i]="$(asset_path "$(manifest_value "/assets/$i/path")")"
    URLS[i]="$(manifest_value "/assets/$i/url")"
    DRIVE_IDS[i]="$(manifest_value "/assets/$i/drive_id")"
    SIZES[i]="$(manifest_value "/assets/$i/size")"
    HASHES[i]="$(manifest_value "/assets/$i/sha256")"
    [[ "${SIZES[i]}" =~ ^[1-9][0-9]*$ && "${HASHES[i]}" =~ ^[a-f0-9]{64}$ && "${DRIVE_IDS[i]}" =~ ^[A-Za-z0-9_-]+$ ]] || fail 'Invalid asset verification information in the manifest.'
    manual "Download ${IDS[i]}
From: ${URLS[i]}
Save exactly as: ${PATHS[i]}
Expected size: ${SIZES[i]} bytes; SHA-256: ${HASHES[i]}"
done
for i in 0 1 2; do
    if [[ -e "${PATHS[i]}" ]]; then
        verify_file "${IDS[i]}" "${PATHS[i]}" "${SIZES[i]}" "${HASHES[i]}" || fail "Existing file has the wrong size or SHA-256: ${PATHS[i]}. Move it aside and use the manual download instructions below."
    else
        mkdir -p -- "$(dirname -- "${PATHS[i]}")"
        say "Downloading ${IDS[i]} (${SIZES[i]} bytes). Keep this window open."
        download_file "${URLS[i]}" "${PATHS[i]}" "${DRIVE_IDS[i]}"
        verify_file "${IDS[i]}" "$DOWNLOADED" "${SIZES[i]}" "${HASHES[i]}" || fail "Downloaded file failed verification: $DOWNLOADED. Obtain the complete original file using the manual link below."
        mv -- "$DOWNLOADED" "${PATHS[i]}"
        cache_moved_file "${IDS[i]}" "${PATHS[i]}" "${HASHES[i]}"
    fi
    [[ "${IDS[i]}" != docker ]] || DOCKER_ARCHIVE="${PATHS[i]}"
done
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    say 'Importing the Docker image (first run only)...'
    docker load --input "$DOCKER_ARCHIVE" || fail 'Docker image import failed. Check Docker storage/free disk space, then try again.'
fi
say 'Checking NVIDIA GPU access inside Docker...'
docker run --rm --pull never --gpus all --entrypoint nvidia-smi "$IMAGE" -L || fail 'Docker cannot access an NVIDIA GPU. Check the NVIDIA driver and NVIDIA Container Toolkit using the link below.'

VERSION="$(manifest_value /desktop_runtime/version)"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'Invalid desktop runtime version.'
RUNTIME_URL="$(manifest_value /desktop_runtime/linux-x64/url)"
RUNTIME_SHA="$(manifest_value /desktop_runtime/linux-x64/sha256)"
RUNTIME_SIZE="$(manifest_value /desktop_runtime/linux-x64/size)"
RUNTIME_ARCHIVE="$(manifest_value /desktop_runtime/linux-x64/archive)"
[[ "$RUNTIME_URL" == https://github.com/electron/electron/releases/download/* && "$RUNTIME_SHA" =~ ^[a-f0-9]{64}$ && "$RUNTIME_SIZE" =~ ^[1-9][0-9]*$ && "$RUNTIME_ARCHIVE" == "electron-v$VERSION-linux-x64.zip" ]] || fail 'The desktop runtime manifest is invalid.'
RUNTIME_ZIP="$CACHE/$RUNTIME_ARCHIVE"
RUNTIME_DIR="$PROJECT/.fast_demo/electron/linux-x64/$VERSION"
manual "Portable desktop runtime: $RUNTIME_URL
Save exactly as: $RUNTIME_ZIP
SHA-256: $RUNTIME_SHA"
if [[ ! -e "$RUNTIME_ZIP" ]]; then
    say 'Downloading the portable desktop interface...'
    download_file "$RUNTIME_URL" "$RUNTIME_ZIP"
    verify_file electron-linux-x64 "$DOWNLOADED" "$RUNTIME_SIZE" "$RUNTIME_SHA" || fail 'Desktop runtime download failed SHA-256 verification.'
    mv -- "$DOWNLOADED" "$RUNTIME_ZIP"
    cache_moved_file electron-linux-x64 "$RUNTIME_ZIP" "$RUNTIME_SHA"
else
    verify_file electron-linux-x64 "$RUNTIME_ZIP" "$RUNTIME_SIZE" "$RUNTIME_SHA" || fail "The desktop runtime ZIP failed verification: $RUNTIME_ZIP. Move it aside and download the correct file."
fi
if [[ ! -x "$RUNTIME_DIR/electron" || ! -f "$RUNTIME_DIR/.ready" || "$(cat "$RUNTIME_DIR/.ready" 2>/dev/null)" != "$RUNTIME_SHA" ]]; then
    [[ ! -e "$RUNTIME_DIR" ]] || fail "The portable desktop runtime is incomplete: $RUNTIME_DIR. Move this folder aside, then rerun the launcher."
    mkdir -p -- "$RUNTIME_DIR"
    safe_unzip "$RUNTIME_ZIP" "$RUNTIME_DIR"
    chmod u+x "$RUNTIME_DIR/electron" "$RUNTIME_DIR/chrome-sandbox" "$RUNTIME_DIR/chrome_crashpad_handler"
    printf '%s\n' "$RUNTIME_SHA" > "$RUNTIME_DIR/.ready"
fi
OUTPUT="$PROJECT/fast_demo_outputs"
mkdir -p -- "$OUTPUT/.runtime-home"
[[ "$PROJECT" != *,* ]] || fail 'Place this demo in a folder whose path does not contain commas.'
TOKEN="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
CONTAINER="levir-fast-demo-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
docker run --detach --name "$CONTAINER" --label levir-fast-demo=true --init --pull never --gpus all --shm-size=4g \
    --user "$(id -u):$(id -g)" \
    --publish '127.0.0.1::8765' --workdir /workspace/levirdetnet \
    --env HOME=/fast_demo_outputs/.runtime-home \
    --env NO_ALBUMENTATIONS_UPDATE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env "LEVIR_DEMO_TOKEN=$TOKEN" --env "LEVIR_DEMO_HOST_OUTPUT=$OUTPUT" \
    --mount "type=bind,source=$PROJECT,target=/workspace/levirdetnet,readonly" \
    --mount "type=bind,source=$OUTPUT,target=/fast_demo_outputs" \
    "$IMAGE" python -u fast_demo/server.py --host 0.0.0.0 --port 8765 --output-root /fast_demo_outputs \
    || fail 'Could not start the demo container. Check the Docker error above.'
ADDRESS="$(docker port "$CONTAINER" 8765/tcp)"
[[ "$ADDRESS" =~ ^127\.0\.0\.1:[0-9]+$ ]] || fail 'Docker did not report a loopback port for the demo.'
URL="http://$ADDRESS"
say 'Waiting for the local demo interface...'
READY=0
for ((attempt=0; attempt<45; attempt++)); do
    if curl --silent --fail --max-time 2 "$URL/health" >/dev/null; then READY=1; break; fi
    [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" == true ]] || break
    sleep 2
done
if [[ $READY -ne 1 ]]; then docker logs --tail 80 "$CONTAINER"; fail 'The demo server did not start. Review the container error above.'; fi
say "Ready. Results are saved in: $OUTPUT"
say 'Opening the desktop application. Keep this terminal open; closing the application stops its own demo container.'
export LEVIR_DEMO_URL="$URL/#token=$TOKEN" LEVIR_DEMO_OUTPUT="$OUTPUT"
if ! "$RUNTIME_DIR/electron" "$PROJECT/fast_demo/desktop"; then
    fail 'The desktop interface exited with an error. Check the error above. Linux may require the system libraries listed by Electron and support for unprivileged user namespaces; this launcher does not disable its sandbox.'
fi
