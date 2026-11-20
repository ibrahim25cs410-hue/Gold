# Installation

Use Python 3.12 and an isolated virtual environment. Install the same CPU Torch build used by Docker and the Python requirements.

    python -m venv .venv
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
    python -m pip install -r requirements.txt
    python -m pip check

Activate the virtual environment before the pip commands. On PowerShell set the following variables before launching. The app reads process variables; the example file is a reference.

    $env:AURA_WORKER='0'
    $env:AURA_STATE_DIR=(Join-Path (Get-Location) 'state')
    $env:HF_HUB_OFFLINE='1'
    python -m uvicorn app:app --host 127.0.0.1 --port 7860

Supply authorized datasets and the trusted original model bundle through private project channels for full inference. A source-only installation may report missing model readiness. The shadow checkpoints require their own pinned manifests and runtime.
