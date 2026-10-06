@echo off
setlocal
set "LEVIR_LAUNCHER=%~f0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$raw=[IO.File]::ReadAllText($env:LEVIR_LAUNCHER); $marker='#'+' POWERSHELL_START'; & ([scriptblock]::Create(($raw -split [regex]::Escape($marker),2)[1]))"
set "LEVIR_EXIT=%ERRORLEVEL%"
if not "%LEVIR_EXIT%"=="0" pause
exit /b %LEVIR_EXIT%
# POWERSHELL_START
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$script:LauncherRoot = Split-Path -Parent $env:LEVIR_LAUNCHER
$script:Project = $null
$script:ContainerName = $null
$script:Manual = New-Object System.Collections.Generic.List[string]
$script:Repository = 'https://github.com/QinzheYang/LEVIRDet-release'
$script:Archive = 'https://codeload.github.com/QinzheYang/LEVIRDet-release/zip/refs/heads/main'
$script:Manual.Add('Project: ' + $script:Repository)
$script:Manual.Add('Download the complete repository ZIP, extract it as levirdetnet-release beside this launcher, and run the launcher again.')

function Say([string]$Message) { Write-Host ('[LEVIRDetNet] ' + $Message) }
function Is-Project([string]$Path) {
    foreach ($relative in @('demo/image_demo.py','mmdet/apis/levir_image_demo.py','configs/_base_/levirdetnet.py','configs/levirdetnet/levirdetnet-30class.py','configs/levirdetnet/levirdetnet-159class.py','fast_demo/server.py','fast_demo/assets.json','fast_demo/desktop/main.cjs','fast_demo/desktop/package.json','fast_demo/desktop/preload.cjs','fast_demo/static/index.html','fast_demo/static/app.js','fast_demo/static/style.css')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $relative) -PathType Leaf)) { return $false }
    }
    return $true
}
function Show-HelpPage([string]$Reason) {
    $path = Join-Path $script:LauncherRoot 'LEVIRDetNet-setup-help.txt'
    $body = "LEVIRDetNet Fast Demo - setup help`r`n`r`n" + $Reason + "`r`n`r`n" + ($script:Manual -join "`r`n`r`n") + "`r`n`r`nAfter completing these steps, run the launcher again. Existing project files are not overwritten."
    try { [IO.File]::WriteAllText($path, $body, (New-Object Text.UTF8Encoding($false))) } catch { }
    try { Add-Type -AssemblyName System.Windows.Forms; [void][Windows.Forms.MessageBox]::Show($Reason + "`r`n`r`nInstructions were saved to:`r`n" + $path, 'LEVIRDetNet setup', [Windows.Forms.MessageBoxButtons]::OK, [Windows.Forms.MessageBoxIcon]::Warning) } catch { }
}
function Run-Curl([string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    & $script:Curl @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Download failed. Check the connection, proxy, or Google Drive download quota; manual download instructions are shown below.' }
}
function Download-File([string]$Url, [string]$Destination, [string]$DriveId = '') {
    $part = $Destination + '.part'
    $cookies = $Destination + '.cookies'
    $urlToGet = $Url
    if ($DriveId) { $urlToGet = 'https://drive.usercontent.google.com/download?id=' + [Uri]::EscapeDataString($DriveId) + '&export=download&authuser=0&confirm=t' }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        $curlArgs = @('--location','--fail','--show-error','--retry','3','--retry-delay','2','--connect-timeout','30','--speed-limit','1024','--speed-time','120','--cookie',$cookies,'--cookie-jar',$cookies,'--output',$part)
        if ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -gt 0) {
            $existing = [IO.File]::OpenRead($part)
            try { $existingBuffer = New-Object byte[] 1024; $existingCount = $existing.Read($existingBuffer,0,$existingBuffer.Length); $existingPrefix = [Text.Encoding]::UTF8.GetString($existingBuffer,0,$existingCount) } finally { $existing.Dispose() }
            if ($existingPrefix -match '(?is)^\s*(<!doctype\s+html|<html|<\?xml)') { Remove-Item -LiteralPath $part -Force } else { $curlArgs += @('--continue-at','-') }
        }
        Run-Curl ($curlArgs + @($urlToGet))
        $stream = [IO.File]::OpenRead($part)
        try { $buffer = New-Object byte[] 1024; $count = $stream.Read($buffer,0,$buffer.Length); $prefix = [Text.Encoding]::UTF8.GetString($buffer,0,$count) } finally { $stream.Dispose() }
        if ($prefix -notmatch '(?is)^\s*(<!doctype\s+html|<html|<\?xml)') { return $part }
        if ((Get-Item -LiteralPath $part).Length -gt 2097152) { throw 'The download returned an unexpected HTML page instead of a model or Docker archive.' }
        $html = [IO.File]::ReadAllText($part)
        if (-not $DriveId) { throw 'The repository download returned an HTML page instead of a ZIP file.' }
        $form = [regex]::Match($html, '(?is)<form\b[^>]*action=["'']([^"'']+)["''][^>]*>(.*?)</form>')
        if (-not $form.Success) { throw 'Google Drive returned a permission, quota, or sign-in page. Download the file manually using the link below.' }
        $action = [System.Net.WebUtility]::HtmlDecode($form.Groups[1].Value)
        $uri = [Uri]$action
        if ($uri.Scheme -ne 'https' -or $uri.Host -notin @('drive.google.com','drive.usercontent.google.com','docs.google.com')) { throw 'Google Drive returned an unexpected confirmation destination.' }
        $query = New-Object System.Collections.Generic.List[string]
        foreach ($inputMatch in [regex]::Matches($form.Groups[2].Value, '(?is)<input\b[^>]*>')) {
            $name = [regex]::Match($inputMatch.Value, '(?is)\bname=["'']([^"'']*)["'']')
            $value = [regex]::Match($inputMatch.Value, '(?is)\bvalue=["'']([^"'']*)["'']')
            if ($name.Success -and $value.Success) { $query.Add([Uri]::EscapeDataString([System.Net.WebUtility]::HtmlDecode($name.Groups[1].Value)) + '=' + [Uri]::EscapeDataString([System.Net.WebUtility]::HtmlDecode($value.Groups[1].Value))) }
        }
        if ($query.Count -eq 0) { throw 'Google Drive did not provide a usable download confirmation. Download manually using the link below.' }
        $urlToGet = $action + $(if ($action.Contains('?')) { '&' } else { '?' }) + ($query -join '&')
        Remove-Item -LiteralPath $part -Force
    }
    throw 'Google Drive confirmation did not produce a file. Use the manual download link below.'
}
function Ensure-Project {
    foreach ($candidate in @($script:LauncherRoot,(Join-Path $script:LauncherRoot 'levirdetnet-release'),(Join-Path $script:LauncherRoot 'LEVIRDet-release-main'))) {
        if (Is-Project $candidate) { return (Get-Item -LiteralPath $candidate).FullName }
    }
    $target = Join-Path $script:LauncherRoot 'levirdetnet-release'
    if (Test-Path -LiteralPath $target) { throw ('The existing project is incomplete: ' + $target + '. Add the missing fast_demo and project files, or move this launcher to a fresh folder. Existing files will not be overwritten.') }
    Say 'Downloading the complete project from GitHub...'
    $zipPath = Join-Path $script:LauncherRoot 'LEVIRDet-release-main.zip'
    $downloaded = Download-File $script:Archive $zipPath
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($downloaded)
    $extract = Join-Path $script:LauncherRoot ('.fast-demo-code-' + [Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Path $extract)
    try {
        $prefix = [IO.Path]::GetFullPath($extract) + [IO.Path]::DirectorySeparatorChar
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace('\','/')
            if ($name.StartsWith('/') -or $name -match '(^|/)\.\.(/|$)|:' -or ((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000)) { throw 'The repository ZIP contains an unsafe path or a symbolic link.' }
            $dest = [IO.Path]::GetFullPath((Join-Path $extract $name))
            if (-not $dest.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'The repository ZIP contains a path outside the extraction directory.' }
        }
    } finally { $zip.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($downloaded,$extract)
    $children = @(Get-ChildItem -LiteralPath $extract -Directory)
    if ($children.Count -ne 1 -or -not (Is-Project $children[0].FullName)) { throw ('The public repository does not yet contain the complete Fast Demo. Update/download the repository from ' + $script:Repository + ' and try again.') }
    Move-Item -LiteralPath $children[0].FullName -Destination $target
    return $target
}
function Asset-Path([string]$Relative) {
    if ($Relative -match '(^|[\\/])\.\.([\\/]|$)|:' -or [IO.Path]::IsPathRooted($Relative)) { throw 'The asset manifest contains an unsafe path.' }
    $result = [IO.Path]::GetFullPath((Join-Path $script:Project $Relative))
    if (-not $result.StartsWith($script:Project + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'The asset destination must be inside the project.' }
    return $result
}
function File-Sha256([string]$Path) {
    # Use .NET directly: Windows PowerShell may inherit a PSModulePath without
    # the script module that provides Get-FileHash. ComputeHash streams the file.
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::OpenRead($Path)
        try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
        finally { $stream.Dispose() }
    } finally { $algorithm.Dispose() }
}
function Verify-Asset($Asset, [string]$Path) {
    $info = Get-Item -LiteralPath $Path
    if ($Asset.PSObject.Properties['size'] -and $info.Length -ne [long]$Asset.size) { return $false }
    $cachePath = Join-Path $script:Cache ($Asset.id + '.json')
    if (Test-Path -LiteralPath $cachePath) {
        try { $cache = Get-Content -LiteralPath $cachePath -Raw | ConvertFrom-Json; if ($cache.path -eq $Path -and $cache.size -eq $info.Length -and $cache.mtime -eq $info.LastWriteTimeUtc.Ticks.ToString() -and $cache.sha256 -eq $Asset.sha256) { return $true } } catch { }
    }
    Say ('Checking SHA-256: ' + $Asset.path + ' (large files take a few minutes on first use)...')
    $hash = File-Sha256 $Path
    if ($hash -ne $Asset.sha256) { return $false }
    @{ path=$Path; size=$info.Length; mtime=$info.LastWriteTimeUtc.Ticks.ToString(); sha256=$hash } | ConvertTo-Json | Set-Content -LiteralPath $cachePath -Encoding UTF8
    return $true
}
function Cache-MovedAsset($Asset, [string]$Path) {
    $info = Get-Item -LiteralPath $Path
    @{ path=$Path; size=$info.Length; mtime=$info.LastWriteTimeUtc.Ticks.ToString(); sha256=$Asset.sha256 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $script:Cache ($Asset.id + '.json')) -Encoding UTF8
}
function Ensure-Desktop($Manifest) {
    $version = $Manifest.desktop_runtime.version
    $runtime = $Manifest.desktop_runtime.'win32-x64'
    if ($version -notmatch '^\d+\.\d+\.\d+$' -or $runtime.sha256 -notmatch '^[a-f0-9]{64}$' -or $runtime.url -notlike 'https://github.com/electron/electron/releases/download/*' -or $runtime.archive -ne ('electron-v' + $version + '-win32-x64.zip')) { throw 'The desktop runtime manifest is invalid.' }
    $archive = Join-Path $script:Cache $runtime.archive
    $asset = [PSCustomObject]@{ id='electron-win32-x64'; path=$runtime.archive; sha256=$runtime.sha256; size=$runtime.size }
    $script:Manual.Add('Portable desktop runtime: ' + $runtime.url + "`r`nSave exactly as: " + $archive + "`r`nSHA-256: " + $runtime.sha256)
    if (Test-Path -LiteralPath $archive) {
        if (-not (Verify-Asset $asset $archive)) { throw ('The desktop runtime ZIP failed verification: ' + $archive + '. Move it aside and download the correct file.') }
    } else {
        Say 'Downloading the portable desktop interface...'
        $part = Download-File $runtime.url $archive
        if (-not (Verify-Asset $asset $part)) { throw 'Desktop runtime download failed SHA-256 verification.' }
        Move-Item -LiteralPath $part -Destination $archive
        Cache-MovedAsset $asset $archive
    }
    $directory = Join-Path $script:Project ('.fast_demo/electron/win32-x64/' + $version)
    $executable = Join-Path $directory 'electron.exe'
    $ready = Join-Path $directory '.ready'
    if ((Test-Path -LiteralPath $executable) -and (Test-Path -LiteralPath $ready) -and ([IO.File]::ReadAllText($ready).Trim() -eq $runtime.sha256)) { return $executable }
    if (Test-Path -LiteralPath $directory) { throw ('The portable desktop runtime is incomplete: ' + $directory + '. Move this folder aside and run the launcher again.') }
    [void](New-Item -ItemType Directory -Force -Path $directory)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace('\','/')
            if ($name.StartsWith('/') -or $name -match '(^|/)\.\.(/|$)|:' -or ((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000)) { throw 'The desktop runtime ZIP contains an unsafe entry.' }
        }
    } finally { $zip.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($archive,$directory)
    if (-not (Test-Path -LiteralPath $executable)) { throw 'The extracted runtime does not contain electron.exe.' }
    [IO.File]::WriteAllText($ready,$runtime.sha256)
    return $executable
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

try {
    Write-Host ''; Write-Host '  LEVIRDetNet Fast Demo  ' -ForegroundColor White -BackgroundColor Black; Write-Host ''
    $curlCommand = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curlCommand) { throw 'Windows curl.exe is required (included in current Windows 10/11). Update Windows or add curl.exe to PATH.' }
    $script:Curl = $curlCommand.Source
    $dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
    if (-not $dockerCommand) {
        foreach ($candidate in @((Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),(Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'))) {
            if (Test-Path -LiteralPath $candidate) { $dockerCommand = Get-Item -LiteralPath $candidate; break }
        }
    }
    $script:Manual.Add('Docker Desktop with WSL 2 and NVIDIA GPU support: https://docs.docker.com/desktop/features/gpu/ . Install/update the NVIDIA Windows driver, open Docker Desktop, accept its terms yourself, and switch to Linux containers.')
    if (-not $dockerCommand) { throw 'Docker Desktop was not found. Install Docker Desktop and enable its WSL 2 backend, then run this launcher again.' }
    $script:Docker = $(if ($dockerCommand -is [System.Management.Automation.ApplicationInfo]) { $dockerCommand.Source } else { $dockerCommand.FullName })
    if (-not (Docker-Ready)) {
        foreach ($candidate in @((Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),(Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'))) {
            if (Test-Path -LiteralPath $candidate) { Say 'Starting Docker Desktop. Accept any Docker license prompt yourself.'; Start-Process -FilePath $candidate -WindowStyle Hidden; break }
        }
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline -and -not (Docker-Ready)) { Start-Sleep -Seconds 3 }
    }
    if (-not (Docker-Ready)) { throw 'Docker is not ready. Open Docker Desktop, finish its setup, wait until the engine is running, and try again.' }
    $os = (Invoke-Docker info --format '{{.OSType}}').Trim()
    if ($os -ne 'linux') { throw 'Switch Docker Desktop to Linux containers before running this demo.' }
    $script:Project = Ensure-Project
    Say ('Project: ' + $script:Project)
    $manifest = Get-Content -LiteralPath (Join-Path $script:Project 'fast_demo/assets.json') -Raw | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or $manifest.docker_image -ne 'levir-train:cuda121-torch231') { throw 'Unsupported Fast Demo asset manifest. Download the latest complete project.' }
    $script:Cache = Join-Path $script:Project '.fast_demo_cache'
    [void](New-Item -ItemType Directory -Force -Path $script:Cache)
    $assets = @($manifest.assets)
    if ($assets.Count -ne 3 -or @($assets | Where-Object { $_.id -in @('docker','weights30','weights159') } | Select-Object -ExpandProperty id -Unique).Count -ne 3) { throw 'The asset manifest must contain the Docker image and both model weights.' }
    foreach ($asset in $assets) {
        $destination = Asset-Path $asset.path
        $script:Manual.Add(('Download ' + $asset.path + "`nFrom: " + $asset.url + "`nSave exactly as: " + $destination + "`nExpected size: " + $asset.size + ' bytes; SHA-256: ' + $asset.sha256))
    }
    foreach ($asset in $assets) {
        if ($asset.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or [long]$asset.size -le 0 -or $asset.drive_id -notmatch '^[a-zA-Z0-9_-]+$') { throw 'The asset manifest has missing or invalid verification information.' }
        $destination = Asset-Path $asset.path
        if (Test-Path -LiteralPath $destination) {
            if (-not (Verify-Asset $asset $destination)) { throw ('This existing file has the wrong size or SHA-256: ' + $destination + '. Move it aside and download the correct file using the instructions below.') }
        } else {
            [void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination))
            Say ('Downloading ' + $asset.path + ' (' + [Math]::Round([long]$asset.size / 1GB,2) + ' GiB). Keep this window open.')
            $part = Download-File $asset.url $destination $asset.drive_id
            if (-not (Verify-Asset $asset $part)) { throw ('Downloaded file failed verification: ' + $part + '. Use the manual link below to obtain the complete original file.') }
            Move-Item -LiteralPath $part -Destination $destination
            Cache-MovedAsset $asset $destination
        }
    }
    Invoke-Docker image inspect $manifest.docker_image *> $null
    if ($script:DockerExitCode -ne 0) {
        Say 'Importing the Docker image (first run only)...'
        $dockerAsset = $assets | Where-Object id -eq 'docker'
        Invoke-Docker load --input (Asset-Path $dockerAsset.path)
        if ($script:DockerExitCode -ne 0) { throw 'Docker image import failed. Check the Docker disk-image location and free space, then run this launcher again.' }
    }
    Say 'Checking NVIDIA GPU access inside Docker...'
    Invoke-Docker run --rm --pull never --gpus all --entrypoint nvidia-smi $manifest.docker_image -L
    if ($script:DockerExitCode -ne 0) { throw 'The container cannot access an NVIDIA GPU. Check the NVIDIA driver and Docker GPU setup at the link below.' }
    $electron = Ensure-Desktop $manifest
    $output = Join-Path $script:Project 'fast_demo_outputs'
    [void](New-Item -ItemType Directory -Force -Path $output)
    $tokenBytes = New-Object byte[] 32; $rng = [Security.Cryptography.RandomNumberGenerator]::Create(); try { $rng.GetBytes($tokenBytes) } finally { $rng.Dispose() }; $token = ($tokenBytes | ForEach-Object { $_.ToString('x2') }) -join ''
    $script:ContainerName = 'levir-fast-demo-' + [Guid]::NewGuid().ToString('N').Substring(0,12)
    $runArgs = @('run','--detach','--name',$script:ContainerName,'--label','levir-fast-demo=true','--init','--pull','never','--gpus','all','--shm-size=4g','--publish','127.0.0.1::8765','--workdir','/workspace/levirdetnet','--env','NO_ALBUMENTATIONS_UPDATE=1','--env','PYTHONDONTWRITEBYTECODE=1','--env',('LEVIR_DEMO_TOKEN=' + $token),'--env',('LEVIR_DEMO_HOST_OUTPUT=' + $output),'--mount',('type=bind,source=' + $script:Project + ',target=/workspace/levirdetnet,readonly'),'--mount',('type=bind,source=' + $output + ',target=/fast_demo_outputs'),$manifest.docker_image,'python','-u','fast_demo/server.py','--host','0.0.0.0','--port','8765','--output-root','/fast_demo_outputs')
    Invoke-Docker @runArgs
    if ($script:DockerExitCode -ne 0) { throw 'Could not start the demo container. Check the Docker error above; project paths containing commas are not supported.' }
    $address = (Invoke-Docker port $script:ContainerName '8765/tcp').Trim()
    if ($address -notmatch '^127\.0\.0\.1:(\d+)$') { throw 'Docker did not report a loopback port for the demo.' }
    $url = 'http://' + $address
    Say 'Waiting for the local demo interface...'
    $ready = $false; $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        try { $response = Invoke-WebRequest -UseBasicParsing -Uri ($url + '/health') -TimeoutSec 2; if ($response.StatusCode -eq 200) { $ready = $true; break } } catch { }
        $running = (Invoke-Docker inspect --format '{{.State.Running}}' $script:ContainerName 2>$null)
        if ($running -ne 'true') { break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { Invoke-Docker logs --tail 80 $script:ContainerName; throw 'The demo server did not start. Review the container error above.' }
    Say ('Ready. Results are saved in: ' + $output)
    Say 'Opening the desktop application. Keep this terminal open; closing the application stops its own demo container.'
    $env:LEVIR_DEMO_URL = $url + '/#token=' + $token
    $env:LEVIR_DEMO_OUTPUT = $output
    $desktop = Start-Process -FilePath $electron -ArgumentList ('"' + (Join-Path $script:Project 'fast_demo/desktop') + '"') -WorkingDirectory $script:Project -PassThru -Wait
    if ($desktop.ExitCode -ne 0) { throw 'The desktop interface exited with an error. Check Windows graphics-driver support and the runtime files.' }
} catch {
    Write-Host ''; Write-Host ('Setup stopped: ' + $_.Exception.Message) -ForegroundColor Red
    foreach ($line in $script:Manual) { Write-Host ''; Write-Host $line }
    Show-HelpPage $_.Exception.Message
    exit 1
} finally {
    if ($script:ContainerName) { Say 'Stopping this demo container...'; Invoke-Docker stop --time 10 $script:ContainerName 2>$null | Out-Null; Invoke-Docker rm $script:ContainerName 2>$null | Out-Null }
}
