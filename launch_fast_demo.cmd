@echo off
setlocal
set "LEVIR_LAUNCHER_FILE=%~f0"
set "LEVIR_LAUNCHER_ROOT=%~dp0"
powershell.exe -NoLogo -NoProfile -STA -WindowStyle Hidden -ExecutionPolicy Bypass -Command "$raw=[IO.File]::ReadAllText($env:LEVIR_LAUNCHER_FILE); $marker='#'+' POWERSHELL_START'; & ([scriptblock]::Create(($raw -split [regex]::Escape($marker),2)[1]))"
exit /b %ERRORLEVEL%
# POWERSHELL_START
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$script:Root = [IO.Path]::GetFullPath($env:LEVIR_LAUNCHER_ROOT)
$env:LEVIR_LAUNCHER_ROOT = $script:Root
$script:Raw = 'https://raw.githubusercontent.com/QinzheYang/LEVIRDet-release/main/'
$script:Version = '44.5.1'
$script:RuntimeHash = '9b382492dcfee91f8f9e92c91f7972550a1b95d2299cac72279dab33a600d7db'
$script:RuntimeSize = 157998329L
$script:RuntimeName = 'electron-v44.5.1-win32-x64.zip'
$script:RuntimeURL = 'https://github.com/electron/electron/releases/download/v44.5.1/electron-v44.5.1-win32-x64.zip'
$script:Form = $null; $script:Cancelled = $false
$script:Manual = New-Object System.Collections.Generic.List[string]
$script:Required = @('fast_demo/assets.json','fast_demo/launcher/package.json','fast_demo/launcher/main.cjs','fast_demo/launcher/preload.cjs','fast_demo/launcher/controller.cjs','fast_demo/launcher/transfers.cjs','fast_demo/launcher/index.html','fast_demo/launcher/renderer.js','fast_demo/launcher/styles.css','fast_demo/launcher/prerequisites.ps1','fast_demo/launcher/prerequisites.sh')
function Is-Launcher([string]$Directory) {
    foreach ($file in $script:Required) { if (-not (Test-Path -LiteralPath (Join-Path $Directory $file) -PathType Leaf)) { return $false } }
    # A previous launcher may have all files but no Hugging Face fallback.
    # Inspect metadata/source as text; never execute downloaded code to select it.
    try {
        $manifest = [IO.File]::ReadAllText((Join-Path $Directory 'fast_demo/assets.json')) | ConvertFrom-Json
        if (-not $manifest.PSObject.Properties['launcher_download_version'] -or [int]$manifest.launcher_download_version -lt 3) { return $false }
        foreach ($file in @('fast_demo/launcher/controller.cjs','fast_demo/launcher/transfers.cjs')) {
            if ([IO.File]::ReadAllText((Join-Path $Directory $file)) -notmatch '\bdownloadWithMirrors\b') { return $false }
        }
    } catch { return $false }
    return $true
}
function Show-Preparing {
    if ($script:Form) { return }
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()
    $script:Form = New-Object Windows.Forms.Form
    $script:Form.Text = 'LEVIRDetNet - Preparing launcher'
    $script:Form.ClientSize = New-Object Drawing.Size(540,174)
    $script:Form.StartPosition = 'CenterScreen'
    $script:Form.FormBorderStyle = 'FixedDialog'; $script:Form.MaximizeBox = $false
    $script:Form.BackColor = [Drawing.Color]::FromArgb(25,28,32)
    $script:Form.ForeColor = [Drawing.Color]::White
    $script:Label = New-Object Windows.Forms.Label
    $script:Label.Location = New-Object Drawing.Point(22,20); $script:Label.Size = New-Object Drawing.Size(495,55)
    $script:Label.Text = 'Preparing the desktop launcher...'; $script:Form.Controls.Add($script:Label)
    $script:Progress = New-Object Windows.Forms.ProgressBar
    $script:Progress.Location = New-Object Drawing.Point(22,85); $script:Progress.Size = New-Object Drawing.Size(495,20)
    $script:Progress.Style = 'Marquee'; $script:Form.Controls.Add($script:Progress)
    $note = New-Object Windows.Forms.Label
    $note.Location = New-Object Drawing.Point(22,119); $note.Size = New-Object Drawing.Size(495,40)
    $note.Text = 'First launch downloads only the desktop interface. Docker and models wait until you choose Download in the launcher.'
    $script:Form.Controls.Add($note)
    $script:Form.Add_FormClosing({ $script:Cancelled = $true })
    $script:Form.Show(); [Windows.Forms.Application]::DoEvents()
}
function Progress-Update([string]$Message,[long]$Received=0,[long]$Total=0) {
    if ($script:Cancelled) { throw 'Launcher preparation was cancelled. Run the CMD again to resume.' }
    Show-Preparing
    $script:Label.Text = $Message
    if ($Total -gt 0) { $script:Progress.Style='Continuous'; $script:Progress.Value=[Math]::Min(100,[int](100.0*$Received/$Total)) }
    else { $script:Progress.Style='Marquee' }
    [Windows.Forms.Application]::DoEvents()
}
function Sha256([string]$File) {
    $hash=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead($File)
    try { return [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $hash.Dispose() }
}
function Fetch([string]$URL,[string]$Destination,[bool]$Resume=$false) {
    $script:Manual.Add('Download: '+$URL+"`r`nSave as: "+$Destination)
    $part=$Destination+'.part'
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Destination))
    [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
    $request=[Net.HttpWebRequest]::Create($URL); $request.Timeout=30000; $request.ReadWriteTimeout=120000; $request.UserAgent='LEVIRDetNet-Launcher/3'
    $offset=0L
    if ($Resume -and (Test-Path -LiteralPath $part)) { $offset=(Get-Item -LiteralPath $part).Length; if($offset -gt 0){$request.AddRange($offset)} }
    Progress-Update ('Downloading '+[IO.Path]::GetFileName($Destination)+'...')
    $response=$request.GetResponse(); $inputStream=$null; $outputStream=$null
    try {
        $append=$offset -gt 0 -and [int]$response.StatusCode -eq 206
        if ($append -and $response.Headers['Content-Range'] -notlike ('bytes '+$offset+'-*')) { throw 'The server returned an invalid partial download range.' }
        if (-not $append) { $offset=0L }
        $total=if($response.ContentLength -gt 0){$response.ContentLength+$offset}else{0}
        $mode=if($append){[IO.FileMode]::Append}else{[IO.FileMode]::Create}
        $inputStream=$response.GetResponseStream(); $outputStream=[IO.File]::Open($part,$mode,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $buffer=New-Object byte[] 262144; $received=$offset
        $watch=[Diagnostics.Stopwatch]::StartNew(); $last=-1.0
        while(($count=$inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
            $outputStream.Write($buffer,0,$count); $received += $count
            $elapsed=$watch.Elapsed.TotalSeconds
            if($elapsed-$last -ge 0.25 -or $received -eq $total) {
                $speed=if($elapsed -gt 0){($received-$offset)/$elapsed/1MB}else{0}
                Progress-Update ([IO.Path]::GetFileName($Destination)+"`r`n"+[Math]::Round($received/1MB,1)+' MB   '+[Math]::Round($speed,1)+' MB/s') $received $total
                $last=$elapsed
            }
        }
        if($total -gt 0 -and $received -ne $total){throw 'The download was interrupted. Run the launcher again to retry.'}
    } finally {if($inputStream){$inputStream.Dispose()};if($outputStream){$outputStream.Dispose()};$response.Dispose()}
    return $part
}
function Ensure-Runtime([string]$Base) {
    $runtime=Join-Path $Base ('.fast_demo/electron/win32-x64/'+$script:Version)
    $executable=Join-Path $runtime 'electron.exe'; $ready=Join-Path $runtime '.ready'
    if((Test-Path -LiteralPath $executable) -and (Test-Path -LiteralPath $ready) -and [IO.File]::ReadAllText($ready).Trim() -eq $script:RuntimeHash){return $executable}
    $cache=Join-Path $Base '.fast_demo_cache'; [void][IO.Directory]::CreateDirectory($cache)
    $archive=Join-Path $cache $script:RuntimeName
    $script:Manual.Add('Desktop runtime: '+$script:RuntimeURL+"`r`nSave as: "+$archive+"`r`nSHA-256: "+$script:RuntimeHash)
    if(-not(Test-Path -LiteralPath $archive)) {
        $part=$archive+'.part'
        if(-not(Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $script:RuntimeSize){$part=Fetch $script:RuntimeURL $archive $true}
        Progress-Update 'Verifying the desktop runtime...'
        if((Get-Item -LiteralPath $part).Length -ne $script:RuntimeSize -or (Sha256 $part) -ne $script:RuntimeHash){throw ('Desktop runtime verification failed. Move the invalid file aside and download it manually: '+$part)}
        Move-Item -LiteralPath $part -Destination $archive
    }
    Progress-Update 'Checking the portable desktop runtime...'
    if((Get-Item -LiteralPath $archive).Length -ne $script:RuntimeSize -or (Sha256 $archive) -ne $script:RuntimeHash){throw ('Desktop runtime ZIP verification failed: '+$archive)}
    if(Test-Path -LiteralPath $runtime){throw ('The runtime directory is incomplete. Move it aside and run again: '+$runtime)}
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip=[IO.Compression.ZipFile]::OpenRead($archive)
    try {
        foreach($entry in $zip.Entries) {
            $name=$entry.FullName.Replace('\','/')
            if($name.StartsWith('/') -or $name -match '(^|/)\.\.(/|$)|:' -or ((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000)){throw 'The desktop runtime ZIP contains an unsafe entry.'}
        }
    } finally {$zip.Dispose()}
    Progress-Update 'Extracting the desktop runtime...'
    [void][IO.Directory]::CreateDirectory($runtime)
    [IO.Compression.ZipFile]::ExtractToDirectory($archive,$runtime)
    [IO.File]::WriteAllText($ready,$script:RuntimeHash)
    return $executable
}
try {
    $architecture=if($env:PROCESSOR_ARCHITEW6432){$env:PROCESSOR_ARCHITEW6432}else{$env:PROCESSOR_ARCHITECTURE}
    if($architecture -ne 'AMD64'){throw 'This desktop launcher requires Windows x64.'}
    $appRoot=$null
    foreach($candidate in @($script:Root,(Join-Path $script:Root 'levirdetnet-release'),(Join-Path $script:Root 'LEVIRDet-release-main'),(Join-Path $script:Root '.levir-launcher-v3'))) {if(Is-Launcher $candidate){$appRoot=$candidate;break}}
    if(-not$appRoot) {
        $appRoot=Join-Path $script:Root '.levir-launcher-v3'
        foreach($file in $script:Required) {
            $destination=Join-Path $appRoot $file
            if(-not(Test-Path -LiteralPath $destination)) {
                $part=Fetch ($script:Raw+$file) $destination
                $text=[IO.File]::ReadAllText($part)
                if($text.Length -lt 2 -or $text -match '(?is)^\s*(<!doctype\s+html|<html)' -and $file -notlike '*.html'){throw ('GitHub returned an unexpected page. Publish the launcher files to main first: '+$file)}
                Move-Item -LiteralPath $part -Destination $destination
            }
        }
        foreach($file in @('fast_demo/launcher/assets/background.png','fast_demo/launcher/assets/logo.png','fast_demo/launcher/assets/wordmark.png')) {
            $destination=Join-Path $appRoot $file
            if(-not(Test-Path -LiteralPath $destination)) {try{$part=Fetch ($script:Raw+$file) $destination;Move-Item -LiteralPath $part -Destination $destination}catch{Write-Warning ('Optional launcher artwork unavailable: '+$file)}}
        }
    }
    if (-not (Is-Launcher $appRoot)) { throw ('The launcher files do not contain the current Hugging Face download support. Publish the updated fast_demo files to GitHub main, then move this incomplete launcher cache aside and reopen the CMD: '+$appRoot) }
    $electron=$null
    foreach($candidate in @($script:Root,(Join-Path $script:Root 'levirdetnet-release'),(Join-Path $script:Root 'LEVIRDet-release-main'),(Join-Path $script:Root '.levir-launcher'),$appRoot)) {
        $exe=Join-Path $candidate ('.fast_demo/electron/win32-x64/'+$script:Version+'/electron.exe');$ready=Join-Path (Split-Path -Parent $exe) '.ready'
        if((Test-Path -LiteralPath $exe) -and (Test-Path -LiteralPath $ready) -and [IO.File]::ReadAllText($ready).Trim() -eq $script:RuntimeHash){$electron=$exe;break}
    }
    if(-not$electron){$electron=Ensure-Runtime $appRoot}
    if($script:Cancelled){throw 'Launcher preparation was cancelled.'}
    if($script:Form){$script:Form.Dispose();$script:Form=$null}
    $process=Start-Process -FilePath $electron -ArgumentList ('"'+(Join-Path $appRoot 'fast_demo/launcher')+'"') -WorkingDirectory $appRoot -PassThru
    exit 0
} catch {
    if($script:Form){$script:Form.Dispose()}
    $help=Join-Path $script:Root 'LEVIRDetNet-launcher-help.txt'
    $body='LEVIRDetNet launcher could not start.'+"`r`n`r`n"+$_.Exception.Message+"`r`n`r`n"+($script:Manual -join "`r`n`r`n")+"`r`n`r`nProject: https://github.com/QinzheYang/LEVIRDet-release`r`nNo Docker image or model was downloaded by this bootstrap."
    try{[IO.File]::WriteAllText($help,$body,(New-Object Text.UTF8Encoding($false)))}catch{}
    try{Add-Type -AssemblyName System.Windows.Forms;[void][Windows.Forms.MessageBox]::Show($body+"`r`n`r`nHelp saved to: "+$help,'LEVIRDetNet launcher')}catch{Write-Host $body}
    exit 1
}
