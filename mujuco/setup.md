# SETUP

## 1. Create a virtual environment

```
cd <your-work-dir>
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Python 3.12+ is required (the Windows patch below uses `delete_on_close`,
added in 3.12).

## 2. Install

CPU-only machine:

```
pip install so101-nexus imageio
```

Machine with an NVIDIA GPU (adds the Warp backend):

```
pip install "so101-nexus[warp]" imageio
```

## 3. GPU only: install CUDA-enabled PyTorch

The Warp environments return observations as torch tensors on the GPU, but the
default PyPI `torch` wheel on Windows is CPU-only. Replace it:

```
pip install torch --index-url https://download.pytorch.org/whl/cu130 --upgrade
```

(~2 GB download. RTX 50-series cards need cu128 or newer; cu130 covers
everything current. Verify with
`python -c "import torch; print(torch.cuda.is_available())"`, which must print `True`.)

## 4. Run the smoke test

```
python smoke_test.py          # all 10 envs (5 CPU + 5 Warp)
python smoke_test.py --cpu    # CPU/MuJoCo envs only
python smoke_test.py --warp   # Warp GPU envs only
```
