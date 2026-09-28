# Move the project's data/ folder (blueprints, checkpoints, bucket tables, logs; git-ignored) to another
# drive and leave a directory junction at the old path, so every script keeps using "data/...".
# Junctions need no administrator rights (file symlinks do). Nothing is deleted until the copy is verified
# file by file and a blueprint loads through the junction.
#
#   powershell -ExecutionPolicy Bypass -File scripts\move_data_to_d.ps1                 # dry run: checks only
#   powershell -ExecutionPolicy Bypass -File scripts\move_data_to_d.ps1 -Go             # copy, verify, swap
#   powershell -ExecutionPolicy Bypass -File scripts\move_data_to_d.ps1 -Go -DeleteOld  # ... and free C:
#
# Run it only when no training, duel, search or AIVAT job is running: they read and write data/.
param(
    [string]$Src = "C:\Project Manchatten\NegativePluriibus\data",
    [string]$Dst = "D:\NegativePluribus\data",
    [switch]$Go,
    [switch]$DeleteOld
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Output ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }

$item = Get-Item -LiteralPath $Src -Force
if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Say "$Src is already a junction -> $($item.Target)."
    $old = "$Src.old_move"
    if ((Test-Path -LiteralPath $old) -and $DeleteOld) { Say "deleting $old ..."; Remove-Item -LiteralPath $old -Recurse -Force; Say "done: C: freed" }
    elseif (Test-Path -LiteralPath $old) { Say "old copy still at $old (re-run with -DeleteOld to free it)" }
    exit 0
}

# 1. No project job may be running (they hold files in data/ open or append to logs). Any interpreter
#    name counts (the main session runs a renamed copy, tools\python_np.exe), and so do the bash
#    queue scripts that start them; this script's own shells are skipped.
$jobs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -notmatch 'move_data_to_d' -and (
        ($_.Name -match '^python' -and $_.CommandLine -match 'scripts[/\\]|negpluribus|webapp|play_slumbot|aivat') -or
        ($_.Name -eq 'bash.exe' -and $_.CommandLine -match 'queue|duels64|train_pot64|scripts[/\\]')
    )
}
if ($jobs) {
    Say "ABORT: project jobs are running:"
    $jobs | ForEach-Object { Say ("  {0} {1}" -f $_.ProcessId, $_.CommandLine.Substring(0, [Math]::Min(160, $_.CommandLine.Length))) }
    exit 2
}

# 2. Space on the target drive.
$files = Get-ChildItem -LiteralPath $Src -Recurse -File -Force
$bytes = ($files | Measure-Object Length -Sum).Sum
$drive = Get-PSDrive -Name ($Dst.Substring(0, 1))
Say ("data: {0} files, {1:N1} GB; free on {2}: {3:N1} GB" -f $files.Count, ($bytes / 1GB), $drive.Name, ($drive.Free / 1GB))
if ($drive.Free -lt $bytes + 5GB) { Say "ABORT: not enough free space on the target drive"; exit 3 }
if ((Test-Path -LiteralPath $Dst) -and (Get-ChildItem -LiteralPath $Dst -Force | Select-Object -First 1)) {
    Say "note: $Dst exists and is not empty; robocopy will bring it up to date (same files are skipped)"
}
if (-not $Go) { Say "dry run OK. Re-run with -Go to copy, verify and swap."; exit 0 }

# 3. Copy (robocopy keeps timestamps; sequential copy suits an HDD).
New-Item -ItemType Directory -Force -Path $Dst | Out-Null
Say "copying ..."
& robocopy $Src $Dst /E /COPY:DAT /DCOPY:T /R:2 /W:5 /NFL /NDL /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Say "ABORT: robocopy failed with code $LASTEXITCODE; data/ untouched"; exit 4 }

# 4. Verify every file: same relative path, same size, same last-write time.
$bad = 0
foreach ($f in $files) {
    $rel = $f.FullName.Substring($Src.Length)
    $g = Get-Item -LiteralPath ($Dst + $rel) -Force -ErrorAction SilentlyContinue
    if (-not $g -or $g.Length -ne $f.Length -or $g.LastWriteTimeUtc -ne $f.LastWriteTimeUtc) { $bad++; if ($bad -le 10) { Say "MISMATCH: $rel" } }
}
if ($bad) { Say "ABORT: $bad files differ; data/ untouched, the copy stays on the target"; exit 5 }
Say "verified: all $($files.Count) files match (size and time)"

# 5. Swap: rename the old folder (fails if any file is still open), then create the junction.
$old = "$Src.old_move"
try { Rename-Item -LiteralPath $Src -NewName (Split-Path $old -Leaf) }
catch { Say "ABORT: cannot rename data/ (a file is open?): $($_.Exception.Message)"; exit 6 }
cmd /c mklink /J "$Src" "$Dst" | Out-Null
if (-not (Test-Path -LiteralPath $Src)) { Rename-Item -LiteralPath $old -NewName (Split-Path $Src -Leaf); Say "ABORT: junction failed; restored data/"; exit 7 }
Say "junction: $Src -> $Dst"

# 6. Smoke test through the junction: list, and load the production bucketer JSON.
$n = (Get-ChildItem -LiteralPath $Src -Recurse -File -Force | Measure-Object).Count
if ($n -ne $files.Count) { Say "WARNING: $n files through the junction, expected $($files.Count)" }
Push-Location (Split-Path $Src -Parent)
python -c "import json; d=json.load(open('data/buckets_hunl200w3_pot16_s0.json')); print('bucketer loads through the junction:', d['kind'], d['n_buckets'])"
$ok = $LASTEXITCODE -eq 0
Pop-Location
if (-not $ok) { Say "WARNING: smoke test failed; old copy kept at $old" ; exit 8 }

# 7. Free the old copy only when asked.
if ($DeleteOld) { Say "deleting $old ..."; Remove-Item -LiteralPath $old -Recurse -Force; Say "done: C: freed" }
else { Say "done. Old copy kept at $old; delete it with -DeleteOld after checking, or by hand." }
