param(
    [ValidateSet('Check','Ensure')][string]$Mode = 'Check',
    [Parameter(Mandatory=$true)][string]$SetupRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$script:LauncherRoot = [IO.Path]::GetFullPath($SetupRoot)
$script:Docker = $null
$script:Manual = New-Object System.Collections.Generic.List[string]
function Emit-Event($Event) { [Console]::Out.WriteLine('LEVIR_EVENT ' + ($Event | ConvertTo-Json -Compress)) }
function Say([string]$Message) { Emit-Event @{phase='prerequisites'; item='System prerequisites'; message=$Message} }
function Download-File([string]$Url, [string]$Destination) {
    $part = $Destination + '.part'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $request = [Net.HttpWebRequest]::Create($Url)
    $request.Timeout = 30000
    $request.ReadWriteTimeout = 120000
    $request.UserAgent = 'LEVIRDetNet-Launcher/2'
    $response = $request.GetResponse()
    $inputStream = $null; $outputStream = $null
    try {
        $total = [long]$response.ContentLength
        $inputStream = $response.GetResponseStream()
        $outputStream = [IO.File]::Open($part,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $buffer = New-Object byte[] 262144
        $received = 0L; $watch = [Diagnostics.Stopwatch]::StartNew(); $last = -1.0
        while (($count = $inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
            $outputStream.Write($buffer,0,$count); $received += $count
            $elapsed = $watch.Elapsed.TotalSeconds
            if ($elapsed - $last -ge 0.5 -or $received -eq $total) {
                $speed = if ($elapsed -gt 0) { $received / $elapsed } else { 0 }
                $eta = if ($total -gt 0 -and $speed -gt 0) { ($total - $received) / $speed } else { $null }
                Emit-Event @{phase='prerequisites'; item='Docker Desktop'; message='Downloading Docker Desktop installer'; received=$received; total=$(if($total -gt 0){$total}else{$null}); speed=$speed; eta=$eta}
                $last = $elapsed
            }
        }
        if ($total -ge 0 -and $received -ne $total) { throw 'The Docker installer download is incomplete. Please retry.' }
    } finally {
        if ($inputStream) { $inputStream.Dispose() }; if ($outputStream) { $outputStream.Dispose() }; $response.Dispose()
    }
    return $part
}
function Invoke-Docker {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$DockerArguments)
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = 1
    & $script:Docker @DockerArguments
    $script:DockerExitCode = $global:LASTEXITCODE
}
function Docker-Ready {
    $ErrorActionPreference = 'Continue'
    Invoke-Docker info --format '{{.OSType}}' 2>$null | Out-Null
    return $script:DockerExitCode -eq 0
}

function Ask-SetupConsent([string]$Title, [string]$Details) {
    # Every system-changing action has its own default-No decision.
    Add-Type -AssemblyName System.Windows.Forms
    $answer = [Windows.Forms.MessageBox]::Show($Details + "`r`n`r`nInstall now? Choose No to stop setup without making this change.", $Title, [Windows.Forms.MessageBoxButtons]::YesNo, [Windows.Forms.MessageBoxIcon]::Question, [Windows.Forms.MessageBoxDefaultButton]::Button2)
    return $answer -eq [Windows.Forms.DialogResult]::Yes
}
function Find-Docker {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in @((Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),(Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'))) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}
function Find-DockerDesktop {
    foreach ($candidate in @((Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),(Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'))) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}
function Read-WslState {
    $path = Join-Path $env:WINDIR 'System32\wsl.exe'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return [PSCustomObject]@{ path=$path; installed=$false; version=$null } }
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = 1
    $raw = ((& $path --version 2>&1 | Out-String) -replace "`0",'')
    $code = $global:LASTEXITCODE
    # wsl output is localized and some console hosts expose its UTF-16 NULs.
    $firstLine = @($raw -split "`r?`n" | Where-Object { $_.Trim() }) | Select-Object -First 1
    $version = $null
    if ($code -eq 0 -and $firstLine -match '(\d+\.\d+\.\d+(?:\.\d+)?)') { $version = [version]$Matches[1] }
    $installed = $null -ne $version
    if (-not $installed) { & $path --status 2>&1 | Out-Null; $installed = $global:LASTEXITCODE -eq 0 }
    return [PSCustomObject]@{ path=$path; installed=$installed; version=$version }
}
function Run-SetupProcess([string]$Executable, [string[]]$Arguments, [bool]$Elevated) {
    $options = @{ FilePath=$Executable; ArgumentList=($Arguments -join ' '); PassThru=$true; Wait=$true; WindowStyle='Normal' }
    if ($Elevated) { $options.Verb = 'RunAs' }
    try { $process = Start-Process @options }
    catch { throw ('The installer could not start or its administrator prompt was cancelled: ' + $_.Exception.Message) }
    return [int]$process.ExitCode
}
function Check-SetupExit([int]$Code, [string]$Component) {
    if ($Code -in @(1641,3010)) { throw ($Component + ' requested a Windows restart. Save your work, restart Windows yourself, and run this launcher again. The launcher does not restart Windows.') }
    if ($Code -ne 0) { throw ($Component + ' setup failed or was cancelled (exit code ' + $Code + '). See the official setup instructions below, then run the launcher again.') }
}
function Ensure-Wsl {
    $state = Read-WslState
    if ($state.version -and $state.version -ge [version]'2.1.5') { return }
    $script:Manual.Add('WSL 2 setup/update: https://learn.microsoft.com/windows/wsl/install . Docker Desktop needs WSL 2.1.5 or newer and hardware virtualization enabled. Restart Windows if requested, then run this launcher again.')
    if (-not (Test-Path -LiteralPath $state.path -PathType Leaf)) { throw 'This Windows installation does not provide wsl.exe. Update Windows and follow the WSL setup instructions below before running the demo.' }
    if ($state.installed) {
        $details = 'WSL is present, but its version is older than 2.1.5 or cannot be read. Update WSL from Microsoft to use Docker Desktop. This downloads and updates the WSL system component, requires an administrator/UAC approval, and may require a restart. Existing Linux distributions are kept. No restart is performed by this launcher.'
        $arguments = @('--update')
    } else {
        $details = 'WSL 2 is missing or its version cannot be read. Install/update Microsoft WSL using wsl --install --no-distribution. This may enable Windows virtualization features, requires an administrator/UAC approval, downloads system components, and may require a restart. It does not install a Linux distribution or restart Windows automatically.'
        $arguments = @('--install','--no-distribution')
    }
    if (-not (Ask-SetupConsent 'LEVIRDetNet - Microsoft WSL' $details)) { throw 'WSL setup was declined. No WSL installation or update was started.' }
    Say 'Running the approved Microsoft WSL setup. Complete the Windows administrator prompt if shown...'
    Check-SetupExit (Run-SetupProcess $state.path $arguments $true) 'WSL'
    $after = Read-WslState
    if (-not $after.version -or $after.version -lt [version]'2.1.5') { throw 'WSL setup finished but WSL 2.1.5+ is not ready. Restart Windows if the installer requested it, finish the official WSL setup, and run this launcher again.' }
}
function Get-InstallerSignature([string]$Path) {
    # Explicit module path also works with an inherited custom PSModulePath.
    Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
    return Get-AuthenticodeSignature -LiteralPath $Path
}
function Assert-DockerInstaller([string]$Path) {
    $signature = Get-InstallerSignature $Path
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate) { throw 'Docker Desktop installer signature is invalid or could not be verified. The installer was not run. Check Windows certificate trust/network access or install manually from Docker.' }
    $publisher = $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName,$false)
    if ($publisher -notin @('Docker Inc','Docker Inc.')) { throw ('The installer is not signed by Docker Inc (publisher: ' + $publisher + '). The installer was not run.') }
}
function Install-DockerDesktop {
    $url = 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe'
    $directory = Join-Path $script:LauncherRoot '.fast_demo_setup'
    $installer = Join-Path $directory 'Docker Desktop Installer.exe'
    $script:Manual.Add('Docker Desktop for Windows: https://docs.docker.com/desktop/setup/install/windows-install/ . Official installer: ' + $url + "`r`nOptional installer cache: " + $installer)
    $details = 'Docker Desktop was not found. Download its Windows x64 installer from Docker, verify its Docker Inc digital signature, and install it for this Windows user with the WSL 2 backend. Docker downloads and disk storage can take several GB. You must review and accept Docker terms in Docker Desktop yourself; this launcher does not accept any license. WSL setup, when needed, has a separate consent prompt. No restart is performed automatically.'
    if (-not (Ask-SetupConsent 'LEVIRDetNet - Docker Desktop' $details)) { throw 'Docker Desktop installation was declined. No Docker installer was downloaded or started.' }
    Ensure-Wsl
    [void](New-Item -ItemType Directory -Force -Path $directory)
    if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
        Say 'Downloading the approved Docker Desktop installer from Docker...'
        $part = Download-File $url $installer
        Move-Item -LiteralPath $part -Destination $installer
    }
    Say 'Verifying the Docker Inc digital signature...'
    Assert-DockerInstaller $installer
    Say 'Installing Docker Desktop. Review any installer or license window yourself...'
    Check-SetupExit (Run-SetupProcess $installer @('install','--user','--backend=wsl-2') $false) 'Docker Desktop'
    $desktop = Find-DockerDesktop
    $script:Docker = $(if ($desktop) { Join-Path (Split-Path -Parent $desktop) 'resources\bin\docker.exe' } else { Find-Docker })
    if ($script:Docker -and -not (Test-Path -LiteralPath $script:Docker -PathType Leaf)) { $script:Docker = $null }
    if (-not $script:Docker) { throw 'The Docker installer completed but docker.exe was not found. Finish Docker Desktop setup, restart Windows if requested, and run this launcher again.' }
}
function Check-NvidiaDriver {
    $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $path = $(if ($command) { $command.Source } else { Join-Path $env:WINDIR 'System32\nvidia-smi.exe' })
    $script:Manual.Add('NVIDIA Windows GPU driver: https://www.nvidia.com/Download/index.aspx . Install a current driver for your supported NVIDIA GPU, restart if requested, and rerun the launcher. GPU drivers are not installed or changed automatically.')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'An NVIDIA GPU driver (nvidia-smi.exe) was not found. This GPU demo requires a supported NVIDIA GPU and Windows driver. Follow the driver instructions below.' }
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = 1
    $result = & $path -L 2>&1
    if ($global:LASTEXITCODE -ne 0 -or ($result | Out-String) -notmatch 'GPU \d+:') { throw 'The NVIDIA Windows driver did not report an available GPU. Check the GPU and driver using the instructions below.' }
}
function Assert-LocalDockerEndpoint {
    $allowed = '^npipe:////\./pipe/(docker_engine|dockerDesktopLinuxEngine)$'
    if ($env:DOCKER_HOST -and $env:DOCKER_HOST -notmatch $allowed) { throw 'DOCKER_HOST selects a remote or unsupported Docker endpoint. Choose your local Docker Desktop context and clear the remote override before starting the demo; the launcher will not change contexts or contact a remote engine.' }
    if (-not $script:Docker) {
        if ($env:DOCKER_CONTEXT -and $env:DOCKER_CONTEXT -notin @('default','desktop-linux')) { throw ('DOCKER_CONTEXT selects ' + $env:DOCKER_CONTEXT + '. Choose a local Docker Desktop context before installing prerequisites; no context is changed automatically.') }
        return
    }
    $endpoint = (Invoke-Docker context inspect --format '{{.Endpoints.docker.Host}}' 2>$null | Out-String).Trim()
    if ($script:DockerExitCode -ne 0 -or $endpoint -notmatch $allowed) { throw ('The selected Docker context is not a supported local Docker Desktop named pipe (' + $endpoint + '). Select the local default or desktop-linux context in Docker Desktop, clear any remote override, and retry. No Docker context was changed.') }
}
function Ensure-WindowsPrerequisites {
    $architecture = $(if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE })
    if ($architecture -ne 'AMD64') { throw 'This launcher and GPU image require Windows x64 on a supported NVIDIA GPU. Windows ARM is not supported.' }
    $script:Docker = Find-Docker
    Assert-LocalDockerEndpoint
    Check-NvidiaDriver
    $script:Manual.Add('Docker Desktop GPU support on Windows: https://docs.docker.com/desktop/features/gpu/ . Use its WSL 2 backend and Linux containers. NVIDIA Container Toolkit is integrated with Docker Desktop GPU support; do not install the Linux toolkit on Windows. Docker Desktop terms must be accepted by you in its own window.')
    # A working Docker engine is left untouched: no upgrades, configuration edits,
    # WSL changes, or Docker service restart are performed on this path.
    if ($script:Docker -and (Docker-Ready)) { return }
    $desktop = Find-DockerDesktop
    if (-not $desktop) { Install-DockerDesktop; $desktop = Find-DockerDesktop }
    else { Ensure-Wsl }
    if (-not $script:Docker) { $script:Docker = Find-Docker }
    if (-not $script:Docker) { throw 'Docker Desktop is installed but docker.exe is missing. Repair Docker Desktop using its official installer, then rerun this launcher.' }
    Assert-LocalDockerEndpoint
    if ($desktop -and -not (Docker-Ready)) { Say 'Starting Docker Desktop. Review and accept its license terms yourself in Docker Desktop.'; Start-Process -FilePath $desktop -WindowStyle Normal }
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline -and -not (Docker-Ready)) { Start-Sleep -Seconds 3 }
    if (-not (Docker-Ready)) { throw 'Docker is not ready. Finish Docker Desktop license/setup prompts, use the WSL 2 backend and Linux containers, and wait until its engine is running. Restart Windows if WSL setup requested it, then rerun this launcher.' }
}

function Check-Prerequisites {
    $details = New-Object System.Collections.Generic.List[string]
    $needsInstall = $false; $driver = $true; $engine = $false
    $architecture = $(if($env:PROCESSOR_ARCHITEW6432){$env:PROCESSOR_ARCHITEW6432}else{$env:PROCESSOR_ARCHITECTURE})
    if ($architecture -ne 'AMD64') { $details.Add('Windows x64 and a supported NVIDIA GPU are required.'); $driver=$false }
    $script:Docker = Find-Docker
    $desktop = Find-DockerDesktop
    if (-not $desktop) { $details.Add('Docker Desktop was not found. Installation requires your consent.'); $needsInstall=$true }
    try { Assert-LocalDockerEndpoint } catch { $details.Add($_.Exception.Message); return @{ok=$false;dockerPath=$script:Docker;needsInstall=$needsInstall;details=@($details.ToArray())} }
    try { Check-NvidiaDriver } catch { $driver=$false; $details.Add($_.Exception.Message) }
    if ($script:Docker) {
        $os = (Invoke-Docker info --format '{{.OSType}}' 2>$null | Out-String).Trim()
        $engine = $script:DockerExitCode -eq 0 -and $os -eq 'linux'
        if (-not $engine) { $details.Add('Docker Desktop is installed but its Linux container engine is not ready.') }
    } else { $details.Add('Docker CLI was not found.') }
    if (-not $engine) {
        $wsl = Read-WslState
        if (-not $wsl.version -or $wsl.version -lt [version]'2.1.5') { $needsInstall=$true; $details.Add('WSL 2.1.5 or newer is required; installation/update may require a restart.') }
    }
    if ($engine -and $driver) { $details.Add('Local Docker Linux engine and NVIDIA driver are available. Windows Docker Desktop provides GPU container support without a separate Toolkit installation.') }
    return @{ok=($engine -and $driver);dockerPath=$script:Docker;needsInstall=$needsInstall;details=@($details.ToArray())}
}
try {
    if ($Mode -eq 'Check') { Check-Prerequisites | ConvertTo-Json -Compress -Depth 5; exit 0 }
    Say 'Checking Docker Desktop, WSL 2, and NVIDIA driver...'
    Ensure-WindowsPrerequisites
    $os = (Invoke-Docker info --format '{{.OSType}}' | Out-String).Trim()
    if ($script:DockerExitCode -ne 0 -or $os -ne 'linux') { throw 'Switch Docker Desktop to Linux containers before continuing.' }
    @{ok=$true;dockerPath=$script:Docker;needsInstall=$false;details=@('Local Docker Desktop Linux engine and NVIDIA driver are ready.')} | ConvertTo-Json -Compress
} catch {
    @{ok=$false;dockerPath=$script:Docker;needsInstall=$true;details=@($_.Exception.Message)+@($script:Manual.ToArray())} | ConvertTo-Json -Compress -Depth 5
    exit 1
}
