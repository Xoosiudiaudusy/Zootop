# Round 7 (GPU_RUN.md): GPU against CPU at EQUAL NUMBERS OF ITERATIONS, 3-max wide, potential-aware exact 64.
# Everything in one unattended run (~5-6 h on the user's PC), resumable: a step whose output exists is skipped.
#
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_round7.ps1            # the real run
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_round7.ps1 -Tiny      # 10-minute rehearsal (tiny numbers)
#
# Needs from round 6: data\eq3\buckets_cpu0.json, blueprint_cpu0.bin, blueprint_cpu1.bin, blueprint_gpu16k.bin
# (767,295,488 iterations) and the bucket table data\eq\bucket_tables.  Writes r7_*.log, round7_progress.log and
# round7_summary.txt in the current directory.
param(
    [string]$D = "data\eq3",
    [string]$Tables = "data\eq\bucket_tables",
    [string]$Py = "python",
    [switch]$Tiny,     # rehearsal: tiny iterations and deals, outputs under $D\tiny
    [switch]$Emulate   # rehearsal without a GPU (the kernels emulated on the CPU)
)
$ErrorActionPreference = "Continue"
$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path $Tables).Path

$Shallow = 79200000   # iterations of cpu0 / cpu1 in round 6
$Deep = 767295488     # iterations of gpu16k in round 6
$Deals = 1000000
$Out = $D
$Prefix = "r7"
if ($Tiny) {
    $Shallow = 16384; $Deep = 32768; $Deals = 300
    $Out = Join-Path $D "tiny"; $Prefix = "r7tiny"
    New-Item -ItemType Directory -Force $Out | Out-Null
}
$GpuFlag = @("--gpu", "0")
if ($Emulate) { $GpuFlag = @("--gpu-emulate") }
$Game = @("--players", "3", "--stack", "100", "--street", "river", "--preflop-fracs", "0.5,1.0,3.0",
          "--postflop-fracs", "0.5,1.0,2.0,4.0", "--max-raises", "3")
$G = $Game + @("--buckets", "64", "--buckets-kind", "potential", "--exact-features", "--backend", "cpp",
               "--eval-deals", "0", "--no-l1", "--data-dir", $Out)

function Log([string]$m) {
    $s = "$(Get-Date -Format s)  $m"
    Write-Host $s
    Add-Content -Path "round7_progress.log" -Value $s
}

# a blueprint of an earlier round (the tiny rehearsal trains small stand-ins instead)
function Bp([string]$tag) {
    if ($Tiny) { return (Join-Path $Out "blueprint_$tag.bin") }
    return (Join-Path $D "blueprint_$tag.bin")
}

function Train([string]$tag, [int]$seed, [long]$iters, [string[]]$extra) {
    $bp = Join-Path $Out "blueprint_$tag.bin"
    if (Test-Path $bp) { Log "train $tag : exists, skipped"; return }
    Copy-Item (Join-Path $D "buckets_cpu0.json") (Join-Path $Out "buckets_$tag.json") -Force
    Log "train $tag : start ($iters iterations, seed $seed $extra)"
    $t0 = Get-Date
    & $Py scripts/train_blueprint.py @G --seed $seed --iters $iters --tag $tag @extra *> "${Prefix}_train_$tag.log"
    $code = $LASTEXITCODE
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0 -or -not (Test-Path $bp)) { Log "train $tag : FAILED (exit $code) after $min min, see ${Prefix}_train_$tag.log" }
    else { Log "train $tag : done in $min min" }
}

# duels run as separate processes, two at a time (each holds two blueprints in memory, ~9 GB)
function DuelPair([object[]]$pairs) {
    $procs = @()
    foreach ($p in $pairs) {
        $name, $a, $b, $la, $lb = $p
        $log = "${Prefix}_duel_$name.log"
        if ((Test-Path $log) -and (Select-String -Path $log -Pattern " vs " -Quiet)) { Log "duel $name : exists, skipped"; continue }
        if (-not (Test-Path $a) -or -not (Test-Path $b)) { Log "duel $name : SKIPPED, missing $a or $b"; continue }
        $args = @("scripts/compare_checkpoints.py") + $Game + @("--buckets", (Join-Path $D "buckets_cpu0.json"),
                  "--deals", "$Deals", "--a", $a, "--b", $b, "--label-a", $la, "--label-b", $lb)
        Log "duel $name : start ($la vs $lb, $Deals deals)"
        $procs += Start-Process -FilePath $Py -ArgumentList $args -NoNewWindow -PassThru `
                   -RedirectStandardOutput $log -RedirectStandardError "${Prefix}_duel_$name.err"
    }
    foreach ($pr in $procs) { $pr.WaitForExit() }
    foreach ($p in $pairs) { Log "duel $($p[0]) : finished" }
}

Log "===== round 7 start (Tiny=$Tiny Emulate=$Emulate) ====="
if ($Tiny) {
    # stand-ins for the round-6 blueprints
    Train "cpu0" 0 $Shallow @()
    Train "cpu1" 1 $Shallow @()
    Train "gpu16k" 0 $Deep ($GpuFlag + @("--batch", "16384"))
}
# 1. equal iterations, shallow (79.2M = cpu0 / cpu1 of round 6)
Train "gpu4k_79M" 0 $Shallow ($GpuFlag + @("--batch", "4096"))
Train "gpu16k_79M" 0 $Shallow ($GpuFlag + @("--batch", "16384"))
Train "gpu16k_79M_s1" 1 $Shallow ($GpuFlag + @("--batch", "16384"))
# 2. equal iterations, deep (767.3M = gpu16k of round 6)
Train "gpu4k_767M" 0 $Deep ($GpuFlag + @("--batch", "4096"))
Train "cpu_767M" 0 $Deep @()
# 3. duels
$o = $Out
DuelPair @(
    @("gpu4k_79M_vs_cpu0", (Join-Path $o "blueprint_gpu4k_79M.bin"), (Bp "cpu0"), "gpu4k_79M", "cpu0"),
    @("gpu16k_79M_vs_cpu0", (Join-Path $o "blueprint_gpu16k_79M.bin"), (Bp "cpu0"), "gpu16k_79M", "cpu0"))
DuelPair @(
    @("gpu16k_79M_s1_vs_cpu1", (Join-Path $o "blueprint_gpu16k_79M_s1.bin"), (Bp "cpu1"), "gpu16k_79M_s1", "cpu1"),
    @("gpu16k_767M_vs_cpu_767M", (Bp "gpu16k"), (Join-Path $o "blueprint_cpu_767M.bin"), "gpu16k_767M", "cpu_767M"))
DuelPair @(
    @("gpu4k_767M_vs_cpu_767M", (Join-Path $o "blueprint_gpu4k_767M.bin"), (Join-Path $o "blueprint_cpu_767M.bin"), "gpu4k_767M", "cpu_767M"),
    @("cpu_767M_vs_cpu0", (Join-Path $o "blueprint_cpu_767M.bin"), (Bp "cpu0"), "cpu_767M", "cpu0"))

# 4. summary
$sum = "${Prefix}_summary.txt"
"round 7 summary $(Get-Date -Format s)" | Set-Content $sum
foreach ($f in Get-ChildItem "${Prefix}_train_*.log") {
    Add-Content $sum "== $($f.Name)"
    Select-String -Path $f.FullName -Pattern "GPU:|iterations in|done in|saved|Error|error" | ForEach-Object { Add-Content $sum $_.Line }
}
foreach ($f in Get-ChildItem "${Prefix}_duel_*.log") {
    Add-Content $sum "== $($f.Name)"
    Select-String -Path $f.FullName -Pattern " vs |infosets" | ForEach-Object { Add-Content $sum $_.Line }
}
Log "===== round 7 done, summary in $sum ====="
