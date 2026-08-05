# SAM 3.1 perception server

This folder is independent from the MuJoCo environment so it can be copied to
a GPU server. It accepts RGB PNG images plus text prompts and returns masks.
It never receives MuJoCo IDs, object poses, region contents, or depth.

## Server setup

SAM 3.1 currently requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+, and access
to Meta's gated checkpoint on Hugging Face. Request access to
`facebook/sam3.1`, then authenticate on the server with `hf auth login`.

Follow the official SAM repository's current CUDA/PyTorch installation command,
then install this service:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git"
uv pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8010
```

Confirm that the checkpoint loaded before running MuJoCo:

```bash
curl http://127.0.0.1:8010/health
# {"ok": true, "model": "sam3.1"}
```

The first launch downloads the SAM 3.1 checkpoint. To avoid that, set
`SAM3_CHECKPOINT` to an already downloaded `sam3.1_multiplex.pt`. Set the same
`SAM3_API_KEY` on the server and simulator machine when the port is reachable
over a network. Prefer SSH port forwarding instead of exposing it publicly:

```bash
ssh -L 8010:127.0.0.1:8010 user@gpu-server
```

## Test without a GPU

This checks HTTP transport and mask encoding only; it is not a perception test:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv run python -m unittest test_server.py
uv run python server.py --contract-test
```

In a second terminal, run the simulator command documented in
`mujoco_scenes/POINT_CLOUD_PERCEPTION.md`. The health endpoint should report
`contract_test_not_sam` in this mode and `sam3.1` with the real model.

To copy only this workspace onto a server from the Git repository:

```bash
git clone --filter=blob:none --no-checkout REPOSITORY_URL sam3-service
cd sam3-service
git sparse-checkout init --cone
git sparse-checkout set perception_server
git checkout main
cd perception_server
```
