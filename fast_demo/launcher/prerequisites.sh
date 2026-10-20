#!/usr/bin/env bash
# Read-only Check or consent-gated Ensure; no model/code/image downloads here.
set -Eeuo pipefail
MODE=Check
LAUNCHER_ROOT=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|-Mode) MODE="$2"; shift 2;;
        --setup-root|-SetupRoot) LAUNCHER_ROOT="$2"; shift 2;;
        *) printf 'Unknown prerequisite argument: %s\n' "$1" >&2; exit 2;;
    esac
done
[[ "$MODE" == Check || "$MODE" == Ensure ]] && [[ -n "$LAUNCHER_ROOT" ]] || { printf 'Use --mode Check|Ensure --setup-root PATH\n' >&2; exit 2; }
json_string() {
    local value="$1"
    value=${value//\\/\\\\}; value=${value//\"/\\\"}; value=${value//$'\n'/\\n}; value=${value//$'\r'/\\r}; value=${value//$'\t'/\\t}
    printf '"%s"' "$value"
}
say() { printf 'LEVIR_EVENT {"phase":"prerequisites","item":"System prerequisites","message":'; json_string "$*"; printf '}\n'; }
manual_links='Docker Engine: https://docs.docker.com/engine/install/
NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Automatic setup supports Ubuntu 22.04/24.04/26.04 or Debian 12/13 on x86-64 with systemd. Use an ordinary desktop account. NVIDIA display drivers must be installed manually.'
fail() {
    trap - ERR
    printf '{"ok":false,"dockerPath":'; json_string "$(command -v docker || true)"; printf ',"needsInstall":true,"details":['; json_string "$*"; printf ','; json_string "$manual_links"; printf ']}\n'
    exit 1
}
trap 'fail "Prerequisite setup stopped at line $LINENO; inspect the installer output above."' ERR
ask() {
    local title="$1" message="$2" reply='' response=0
    say "$title: $message"
    if command -v zenity >/dev/null 2>&1; then
        zenity --question --default-cancel --title="LEVIRDetNet - $title" --width=620 --ok-label=Yes --cancel-label=No --text="$message" 2>/dev/null
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --title "LEVIRDetNet - $title" --defaultno --yesno "$message" 2>/dev/null
    elif command -v xmessage >/dev/null 2>&1; then
        xmessage -center -title "LEVIRDetNet - $title" -buttons Yes:42,No:1 -default No "$message" 2>/dev/null || response=$?
        [[ $response -eq 42 ]]
    elif [[ -t 0 && -t 1 ]]; then
        printf 'Continue? Type yes to agree [yes/No]: '
        IFS= read -r reply || return 1
        [[ "$reply" == yes || "$reply" == YES ]]
    else
        say 'No confirmation dialog is available. Run this launcher in a terminal to answer yes/No. Nothing was installed.'
        return 1
    fi
}
root_bash() {
    local body="$1"; shift
    say "Running the approved system setup; administrator authentication may be required."
    # -p and a fixed PATH prevent importing the desktop user's Bash startup code
    # or executable search path into a privileged installation process.
    body=$'set -Eeuo pipefail\nexport PATH=/usr/sbin:/usr/bin:/sbin:/bin\nexport DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l\n'"$body"
    if command -v pkexec >/dev/null 2>&1; then
        pkexec /bin/bash -p -c "$body" levir-setup "$@"
    elif [[ -t 0 && -t 1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo /bin/bash -p -c "$body" levir-setup "$@"
    else
        fail 'Administrator authentication is unavailable. Run this launcher in a terminal with sudo available, or install the prerequisites manually. Do not run the desktop demo as root.'
    fi
}
supported_apt() {
    local ID='' VERSION_ID=''
    [[ -r /etc/os-release ]] || fail 'The Linux distribution could not be identified. Install the missing prerequisite manually.'
    . /etc/os-release
    case "$ID:$VERSION_ID" in
        ubuntu:22.04) DISTRO=ubuntu; CODENAME=jammy;;
        ubuntu:24.04) DISTRO=ubuntu; CODENAME=noble;;
        ubuntu:26.04) DISTRO=ubuntu; CODENAME=resolute;;
        debian:12) DISTRO=debian; CODENAME=bookworm;;
        debian:13) DISTRO=debian; CODENAME=trixie;;
        *) fail "Automatic installation is not supported on $ID $VERSION_ID. Follow the official installation links in the setup help, then reopen the launcher.";;
    esac
    command -v apt-get >/dev/null 2>&1 && command -v dpkg-query >/dev/null 2>&1 || fail 'The supported apt package manager was not found. Install the prerequisites manually.'
}
local_docker_only() {
    local context endpoint security os
    [[ -z "${DOCKER_HOST:-}${DOCKER_CONTEXT:-}${DOCKER_TLS_VERIFY:-}${DOCKER_CERT_PATH:-}" ]] || fail 'A Docker connection override is set. Automatic setup and local file mounting require the local native default Docker daemon. Unset DOCKER_HOST/DOCKER_CONTEXT/TLS overrides or configure and run this demo manually.'
    if command -v docker >/dev/null 2>&1; then
        context="$(docker context show)" || fail 'Docker context detection failed. Repair the Docker CLI manually.'
        [[ "$context" == default ]] || fail "Docker context '$context' is not the local native default. Select a native Docker Engine default context after configuring it manually. Remote, rootless, and Linux Docker Desktop contexts are not changed by this launcher."
        endpoint="$(docker context inspect default --format '{{.Endpoints.docker.Host}}')" || fail 'Could not inspect the default Docker context.'
        [[ "$endpoint" == unix:///var/run/docker.sock || "$endpoint" == unix:///run/docker.sock ]] || fail "The default context points to $endpoint. Only the native local Docker socket is supported for automatic setup."
        if docker info >/dev/null 2>&1; then
            security="$(docker info --format '{{json .SecurityOptions}}')"
            os="$(docker info --format '{{.OperatingSystem}}')"
            [[ "$security" != *rootless* && "$os" != *'Docker Desktop'* ]] || fail 'This daemon is rootless or Docker Desktop for Linux. Configure native Docker Engine with NVIDIA GPU support manually; this launcher will not change that daemon.'
        fi
    fi
}
ensure_download_tools() {
    local program missing=''
    for program in awk sha256sum stat head grep mktemp od tr id wc; do
        command -v "$program" >/dev/null 2>&1 || fail "Missing base Linux utility: $program. Install your distribution's standard core utilities before reopening the launcher."
    done
    for program in curl unzip; do
        command -v "$program" >/dev/null 2>&1 || missing+=" $program"
    done
    [[ -n "$missing" ]] || return 0
    supported_apt
    ask 'Install download tools' "Missing:$missing. Install curl, unzip, and CA certificates using the system apt repositories? This needs administrator authentication and may update dependencies or restart affected system services. No existing packages will be removed." || fail 'Download-tool installation was declined. Install curl and unzip manually, then reopen the launcher.'
    root_bash 'apt-get update
apt-get install -y --no-remove ca-certificates curl unzip' || fail 'Could not install the download tools. Review the package-manager error and follow the manual instructions.'
}
install_docker_if_missing() {
    local conflict status
    command -v docker >/dev/null 2>&1 && return 0
    supported_apt
    [[ ! -S /var/run/docker.sock && ! -S /run/docker.sock ]] || fail 'A Docker socket already exists but the Docker CLI is missing. Repair that installation manually; it will not be replaced.'
    for conflict in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc docker-ce containerd.io; do
        status="$(dpkg-query -W -f='${Status}' "$conflict" 2>/dev/null || true)"
        [[ "$status" != 'install ok installed' ]] || fail "An existing/conflicting package ($conflict) was found. Install or repair Docker manually without replacing your current container stack. This launcher does not uninstall packages."
    done
    command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]] || fail 'Automatic Docker installation requires a systemd Linux desktop. Follow the manual Docker Engine instructions.'
    systemctl is-active --quiet docker && fail 'A Docker service is already active but its CLI is missing. Repair it manually.'
    ask 'Install Docker Engine' "Docker Engine is missing. Install Docker Engine, its CLI, containerd, and Compose/Buildx from Docker's official signed apt repository for $DISTRO $CODENAME? Administrator authentication is required. This adds an apt source and starts Docker's system service; package dependencies may be updated. No existing packages will be removed. Docker Desktop is not installed on Linux." || fail 'Docker installation was declined. Follow the Docker Engine link in the setup help when ready.'
    root_bash 'distro="$1"; codename="$2"
apt-get update
apt-get install -y --no-remove ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
key=/etc/apt/keyrings/levirdet-docker.asc
source=/etc/apt/sources.list.d/levirdet-docker.sources
# Existing repository/key files are not silently replaced.
[[ ! -e "$key" && ! -e "$source" ]] || { echo "LEVIRDetNet apt repository files already exist. Review them and complete Docker installation manually." >&2; exit 1; }
curl --fail --show-error --location "https://download.docker.com/linux/$distro/gpg" -o "$key"
chmod 0644 "$key"
printf "Types: deb\nURIs: https://download.docker.com/linux/%s\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: %s\n" "$distro" "$codename" "$(dpkg --print-architecture)" "$key" > "$source"
apt-get update
apt-get install -y --no-remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl start docker' "$DISTRO" "$CODENAME" || fail 'Docker installation did not complete. Existing packages were not removed; inspect the package-manager error and follow the manual Docker Engine instructions.'
    command -v docker >/dev/null 2>&1 || fail 'Docker was installed but its CLI is still unavailable. Log out and back in, then reopen the launcher.'
}
ensure_docker_access() {
    local account
    docker info >/dev/null 2>&1 && return 0
    if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active --quiet docker; then
        ask 'Start Docker service' 'The native Docker service is not running. Start it now using administrator authentication? Existing containers with restart policies may also start.' || fail 'Docker startup was declined. Start the native Docker service manually and reopen the launcher.'
        root_bash 'systemctl start docker' || fail 'Docker could not be started. Check its system service logs.'
    fi
    docker info >/dev/null 2>&1 && return 0
    if [[ -S /var/run/docker.sock || -S /run/docker.sock ]]; then
        account="$(id -un)"
        if [[ " $(id -nG "$account") " == *' docker '* && " $(id -nG) " != *' docker '* ]]; then
            fail 'Docker group membership is already configured but is not active in this desktop session. Fully log out and log back in, then reopen the launcher.'
        fi
        if [[ " $(id -nG "$account") " != *' docker '* ]]; then
            ask 'Allow this user to use Docker' "Add '$account' to the docker group? This grants root-equivalent control of the computer through Docker. Agree only if you trust this local account. Administrator authentication is required. You must fully log out and log back in, then reopen this launcher to continue setup. The socket permissions will not be weakened." || fail 'Docker group access was declined. Ask your administrator to configure Docker access; the desktop application must run as an ordinary user.'
            root_bash 'getent group docker >/dev/null || groupadd docker
usermod -aG docker -- "$1"' "$account" || fail 'Could not configure Docker group membership. Ask your administrator to configure access.'
            fail 'Docker group membership was added. Fully log out and log back in, then reopen this launcher; remaining prerequisites will be checked automatically.'
        fi
    fi
    fail 'Docker is not accessible to this user. Check its service/socket permissions manually. The launcher will not change socket permissions or run the desktop application as root.'
}
running_containers() {
    local ids
    ids="$(docker ps --quiet)" || fail 'Could not count running Docker containers. No GPU-runtime changes were made.'
    printf '%s\n' "$ids" | awk 'NF {n++} END {print n+0}'
}
ensure_nvidia_toolkit() {
    local runtimes count
    runtimes="$(docker info --format '{{json .Runtimes}}')"
    if [[ "$runtimes" == *'"nvidia"'* ]]; then
        say 'NVIDIA Docker runtime is already configured; leaving the existing setup unchanged.'
        return 0
    fi
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
        supported_apt
        count="$(running_containers)"
        ask 'Install NVIDIA Container Toolkit' "NVIDIA Container Toolkit is missing. Install it using NVIDIA's official signed apt repository? Administrator authentication is required. This adds an apt source and may update dependencies. Running Docker containers: $count. Package installation may restart services and interrupt these containers. Approve only when this interruption is acceptable. No packages will be removed." || fail 'NVIDIA Container Toolkit installation was declined. Follow the NVIDIA setup link when ready.'
        root_bash 'apt-get update
apt-get install -y --no-remove ca-certificates curl gnupg
key=/usr/share/keyrings/levirdet-nvidia-container-toolkit.gpg
source=/etc/apt/sources.list.d/levirdet-nvidia-container-toolkit.list
[[ ! -e "$key" && ! -e "$source" ]] || { echo "LEVIRDetNet NVIDIA repository files already exist. Review them and complete installation manually." >&2; exit 1; }
temp=$(mktemp -d)
trap '\''rm -f -- "$temp/key.asc" "$temp/repository.list"; rmdir -- "$temp"'\'' EXIT
curl --fail --show-error --location https://nvidia.github.io/libnvidia-container/gpgkey -o "$temp/key.asc"
gpg --batch --dearmor --output "$key" "$temp/key.asc"
chmod 0644 "$key"
curl --fail --show-error --location https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$temp/repository.list"
sed "s#deb https://#deb [signed-by=$key] https://#g" "$temp/repository.list" > "$source"
apt-get update
apt-get install -y --no-remove nvidia-container-toolkit' || fail 'NVIDIA Container Toolkit installation did not complete. Review the package-manager error and follow the NVIDIA setup instructions.'
    fi
    command -v nvidia-ctk >/dev/null 2>&1 || fail 'nvidia-ctk is still unavailable. Complete NVIDIA Container Toolkit installation manually.'
    command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]] || fail 'Automatic GPU-runtime configuration requires systemd. Follow the NVIDIA configuration instructions for this Docker Engine manually.'
    ask 'Configure NVIDIA Docker runtime' 'Configure NVIDIA GPU support in /etc/docker/daemon.json? Existing settings will be preserved by nvidia-ctk, and an existing daemon.json will be backed up beside the original file. Administrator authentication is required. A separate restart confirmation follows before any settings change.' || fail 'GPU-runtime configuration was declined. Follow the NVIDIA manual instructions when ready.'
    count="$(running_containers)"
    ask 'Restart Docker to enable GPU support' "Restart the native Docker service after applying the GPU configuration? Running Docker containers: $count. Restarting can interrupt every running container, including work unrelated to this demo. Choose No to postpone; no daemon settings will be changed." || fail 'Docker restart was declined. GPU configuration was left unchanged; reopen this launcher when a service restart is acceptable.'
    local_docker_only
    root_bash 'config=/etc/docker/daemon.json
[[ ! -L /etc/docker && ! -L "$config" ]] || { echo "Symlinked Docker configuration requires manual setup." >&2; exit 1; }
install -d -m 0755 /etc/docker
candidate=$(mktemp /etc/docker/.levir-daemon.XXXXXXXX.json)
trap '\''rm -f -- "$candidate"'\'' EXIT
backup=""
if [[ -e "$config" ]]; then
    backup="$config.levir-backup.$(date -u +%Y%m%dT%H%M%SZ).$$"
    cp -a -- "$config" "$backup"
    cp -p -- "$config" "$candidate"
    printf "Docker configuration backup: %s\n" "$backup"
else
    printf "{}\n" > "$candidate"
    chmod 0644 "$candidate"
fi
nvidia-ctk runtime configure --runtime=docker --config "$candidate"
dockerd --validate --config-file "$candidate"
if [[ -n "$backup" ]]; then
    chmod --reference="$backup" "$candidate"
    chown --reference="$backup" "$candidate"
fi
mv -- "$candidate" "$config"
if ! systemctl restart docker; then
    echo "Docker restart failed; restoring the previous daemon configuration." >&2
    if [[ -n "$backup" ]]; then cp -p -- "$backup" "$config"; else rm -f -- "$config"; fi
    systemctl start docker || true
    exit 1
fi' || fail 'NVIDIA Docker configuration failed. Review the error and any daemon.json backup path above; use the official NVIDIA instructions to complete setup.'
    docker info >/dev/null 2>&1 || fail 'Docker is not ready after the restart. Wait a moment and reopen the launcher.'
    runtimes="$(docker info --format '{{json .Runtimes}}')"
    [[ "$runtimes" == *'"nvidia"'* ]] || fail 'Docker did not report the NVIDIA runtime after configuration. Review the Docker service logs and NVIDIA setup instructions.'
}
ensure_linux_prerequisites() {
    local_docker_only
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || fail 'A working NVIDIA GPU driver was not found. Install the appropriate NVIDIA driver using your distribution or GPU vendor instructions, reboot if requested, and reopen this launcher. GPU drivers are not installed automatically.'
    ensure_download_tools
    install_docker_if_missing
    local_docker_only
    ensure_docker_access
    [[ "$(docker info --format '{{.OSType}}')" == linux ]] || fail 'Docker must use Linux containers.'
    ensure_nvidia_toolkit
}

check_prerequisites() {
    local okay=true install=false docker_path='' details='' endpoint='' context='' runtimes=''
    docker_path="$(command -v docker || true)"
    if [[ -z "$docker_path" ]]; then okay=false; install=true; details+='Docker Engine is not installed. '; fi
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then okay=false; details+='An NVIDIA GPU driver is missing or unavailable; install it manually. '; fi
    if [[ -n "${DOCKER_HOST:-}${DOCKER_CONTEXT:-}${DOCKER_TLS_VERIFY:-}${DOCKER_CERT_PATH:-}" ]]; then
        okay=false; details+='Docker connection overrides must be cleared for the local native Docker daemon. '
    elif [[ -n "$docker_path" ]]; then
        context="$(docker context show 2>/dev/null || true)"
        endpoint="$(docker context inspect default --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)"
        if [[ "$context" != default || ( "$endpoint" != unix:///var/run/docker.sock && "$endpoint" != unix:///run/docker.sock ) ]]; then
            okay=false; details+='Select the native local default Docker context. '
        elif ! docker info >/dev/null 2>&1; then okay=false; details+='Docker is stopped or this user lacks access. '
        else
            if [[ "$(docker info --format '{{.OSType}}')" != linux || "$(docker info --format '{{.OperatingSystem}}')" == *'Docker Desktop'* || "$(docker info --format '{{json .SecurityOptions}}')" == *rootless* ]]; then
                okay=false; details+='Use native local Docker Engine with Linux containers (not Linux Docker Desktop or rootless Docker). '
            fi
            runtimes="$(docker info --format '{{json .Runtimes}}')"
            if [[ "$runtimes" != *'"nvidia"'* ]]; then okay=false; install=true; details+='NVIDIA Container Toolkit installation/configuration is required. '; fi
        fi
    fi
    [[ "$okay" != true ]] || details='Local Docker Engine, NVIDIA driver, and GPU runtime are available.'
    printf '{"ok":%s,"dockerPath":' "$okay"; json_string "$docker_path"; printf ',"needsInstall":%s,"details":[' "$install"; json_string "$details"; printf ']}\n'
}
if [[ "$MODE" == Check ]]; then check_prerequisites; exit 0; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This launcher requires Linux x86-64.'
[[ $EUID -ne 0 ]] || fail 'Run the launcher as an ordinary desktop user; administrator authentication is requested only for approved installation steps.'
say 'Checking Docker Engine, NVIDIA driver, and NVIDIA Container Toolkit...'
ensure_linux_prerequisites
printf '{"ok":true,"dockerPath":'; json_string "$(command -v docker)"; printf ',"needsInstall":false,"details":["Local Docker Engine, NVIDIA driver, and GPU runtime are ready."]}\n'
