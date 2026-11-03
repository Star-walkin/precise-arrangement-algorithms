$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $Root
$BasePython = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Sam2 = Join-Path $Root 'vendor\sam2'
$Checkpoint = Join-Path $Root 'checkpoints\sam2.1_hiera_tiny.pt'
$CheckpointSha256 = '7402E0D864FA82708A20FBD15BC84245C2F26DFF0EB43A4B5B93452DEB34BE69'

function Get-CheckpointSha256 {
    if (-not (Test-Path -LiteralPath $Checkpoint)) { return '' }
    return (& $Python -c "import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1],'rb'),'sha256').hexdigest().upper())" $Checkpoint).Trim()
}

if (-not (Test-Path -LiteralPath $Python)) {
    & $BasePython -m venv (Join-Path $Root '.venv')
}
& $Python -m pip install --upgrade pip

# The first six anchors use the 120 MB CPU build.  The CUDA wheel is about
# 2.6 GB and belongs to the later training/runtime environment, not this
# small human-in-the-loop annotation utility.
& $Python -m pip install torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu

if (-not (Test-Path -LiteralPath (Join-Path $Sam2 '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Sam2) | Out-Null
    git clone --depth 1 https://github.com/facebookresearch/sam2.git $Sam2
}
$env:SAM2_BUILD_CUDA = '0'
& $Python -m pip install --no-build-isolation -e $Sam2

if ((Test-Path -LiteralPath $Checkpoint) -and
    ((Get-CheckpointSha256) -ne $CheckpointSha256)) {
    Remove-Item -LiteralPath $Checkpoint -Force
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Checkpoint) | Out-Null
    & curl.exe -L --retry 3 -o $Checkpoint 'https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt?download=true'
}
if ((Get-CheckpointSha256) -ne $CheckpointSha256) {
    throw 'SAM 2 checkpoint checksum failed. Run the installer again.'
}

& $Python (Join-Path $Root 'prepare_sam2_tasks.py')
& $Python -c "import torch; print('PyTorch', torch.__version__, 'annotation_device=CPU')"
Write-Host ''
Write-Host 'SAM 2 annotation environment is ready.' -ForegroundColor Green
