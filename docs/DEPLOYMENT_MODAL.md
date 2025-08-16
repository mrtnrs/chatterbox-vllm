# Chatterbox-vLLM on Modal: Findings, Fixes, and Checklist

This document tracks the issues encountered when deploying `app.py` to Modal and the working resolutions. Keep it short and actionable.

## Environment summary
- Entrypoint: `app.py` (Modal ASGI app)
- Local source: `src/chatterbox_vllm/`
- vLLM: `0.10.0`
- PyTorch: `2.8.0+cu128`
- Model assets pulled into `t3-model/` at runtime

## Model/tokenizer artifacts created in `t3-model/`
- `t3_cfg.safetensors`, `ve.safetensors`, `s3gen.safetensors`, `conds.pt`
- `tokenizer.json` (downloaded)
- `tokenizer.py` (copied from `src/chatterbox_vllm/models/t3/entokenizer.py`)
- `tokenizer_config.json` with:
  - `{"tokenizer_class": "EnTokenizer", "trust_remote_code": true}`
- `config.json` with `auto_map` pointing to `ChatterboxT3`

## Issues and resolutions

- __vLLM received a tokenizer object (AttributeError: .lower)__
  - Symptom: vLLM calls `.lower()` on `tokenizer`, receiving an `EnTokenizer` instance → crash.
  - Root cause: Passing a tokenizer instance to vLLM. vLLM expects a string path (HF repo ID or local dir).
  - Fix: In `app.py` call `ChatterboxTTS.from_local(ckpt_dir=MODEL_DIR)` (do not pass the object). Current `from_local()` passes `tokenizer=str(ckpt_dir)` to vLLM.

- __EnTokenizer.__init__() missing required `vocab_file`__
  - Symptom: Trace shows `/usr/local/lib/python3.12/site-packages/chatterbox_vllm/tts.py` attempting `EnTokenizer()` with no args.
  - Root cause: The imported `chatterbox_vllm` is the installed package version (not local), with older logic that instantiates `EnTokenizer` directly.
  - Fix options:
    - Recommended: Ensure the local source is imported first.
      - Add near the top of `app.py` (before importing `chatterbox_vllm`):
        ```python
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
        ```
      - This ensures `import chatterbox_vllm` resolves to `src/chatterbox_vllm/` (which passes a path to vLLM).
      - Optional sanity check log:
        ```python
        import chatterbox_vllm as _cb; log.info(f"Using chatterbox_vllm from: {_cb.__file__}")
        ```
    - Or: Pin the exact Git revision that contains the fix.
      - Use proper VCS pinning in `.pip_install(...)`:
        - Example: `"git+https://github.com/mrtnrs/chatterbox-vllm@<commit-sha-or-branch>"`
        - Note: `#6749` is not a valid revision spec. Use `@`.
      - Ensure that revision matches the code where `from_local()` passes a path string to vLLM.

- __Tokenizer files must be present__
  - Ensure `t3-model/tokenizer.json` and `t3-model/tokenizer.py` exist before model init (current `app.py` already does this).
  - `tokenizer_config.json` must declare `tokenizer_class` and `trust_remote_code`.

- __Custom model wiring__
  - Env var set: `VLLM_MODEL_IMPLEMENTATION=chatterbox_vllm.models.t3.modules.t3.ChatterboxT3`.
  - `from_local()` also passes `model_loader_extra_config.model_class` to vLLM.

## Expected logs (happy path)
- "MODEL_DIR: /root/t3-model"
- Downloads for safetensors and `tokenizer.json`
- "Loading tokenizer from: t3-model/tokenizer.json" and test encode succeeds
- `chatterbox_vllm.tts` logs:
  - "Initialized T3Config"
  - "Loaded t3_cfg.safetensors"
  - "Loaded and initialized T3CondEnc / t3_speech_emb / t3_speech_pos_emb"
  - "Calculated vLLM memory: ..."
  - "Using model_path=/root/t3-model, tokenizer_path=/root/t3-model"
  - "Initialized vLLM LLM"
- First request will be slow due to downloads and GPU init

## Modal image tips
- Prefer importing local `src/` to avoid mismatch with installed package. The order of `sys.path` decides.
- If you want to install from GitHub instead, pin with `@branch-or-sha` and remove local `src/` to avoid ambiguity.
- GPU memory utilization is computed dynamically in `from_local()`; you can tune via `max_batch_size`, `max_model_len`, or override `gpu_memory_utilization` in args.

## Quick checklist
- __Imports use local code__: `log __file__` of `chatterbox_vllm` points to `.../src/chatterbox_vllm/__init__.py`.
- __Tokenizer files exist__: `t3-model/tokenizer.json` and `t3-model/tokenizer.py`.
- __tokenizer_config.json correct__: includes `EnTokenizer` and `trust_remote_code`.
- __No tokenizer object passed to vLLM__: only pass the directory path.
- __Custom model wired__: env var + `model_loader_extra_config`.
- __First call warm-up__: expect tens of seconds on cold start.
