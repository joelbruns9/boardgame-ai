## Codex Environment Note

In Codex, project Python/Torch/CUDA commands must be run with escalated permissions.

Use:
`.\.venv\Scripts\python.exe ...`

Known-good verification:
`.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"`

Sandboxed launches may fail because the venv launcher/base Python path is blocked or appears broken from the tool environment.