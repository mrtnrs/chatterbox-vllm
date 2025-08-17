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
- `tokenizer.py` (written at runtime with fresh `EnTokenizer` implementation; ensures correct `from_pretrained` signature)
- `t3.py` shim (re-exports `ChatterboxT3` for dynamic loader: `from chatterbox_vllm.models.t3.modules.t3 import ChatterboxT3`)
- `tokenizer_config.json` with:
  - `{"auto_map": {"AutoTokenizer": ["tokenizer.EnTokenizer", "tokenizer.EnTokenizer"]}, "tokenizer_file": "tokenizer.json", "use_fast": true, "trust_remote_code": true}`
- `config.json` with:
  - `{"auto_map": {"AutoModel": "t3.ChatterboxT3", "AutoTokenizer": ["tokenizer.EnTokenizer", "tokenizer.EnTokenizer"]}, "tokenizer_file": "tokenizer.json", "use_fast": true}`
  - Note: do NOT set `tokenizer_class` here; rely on `auto_map` + `tokenizer_file`.

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

- __Tokenizer files & config must be present and HF-resolvable__
  - Ensure `t3-model/tokenizer.json` and `t3-model/tokenizer.py` exist before model init (current `app.py` already does this).
  - Do NOT set `tokenizer_class` in configs. Instead:
    - In both `config.json` and `tokenizer_config.json` set:
      - `"auto_map": { "AutoTokenizer": ["tokenizer.EnTokenizer", "tokenizer.EnTokenizer"] }`
      - `"tokenizer_file": "tokenizer.json"`
      - `"use_fast": true`
    - Keep `trust_remote_code: true`.

- __Custom model wiring__
  - Env var set: `VLLM_MODEL_IMPLEMENTATION=t3.ChatterboxT3`.
  - `from_local()` also passes `model_loader_extra_config.model_class` to vLLM.
  - A lightweight `t3.py` shim is written into `t3-model/` to back this import path.

## All errors encountered to date (with fixes/status)

- __vLLM `.lower()` on tokenizer object__
  - Symptom: `AttributeError` when vLLM calls `.lower()` on the `tokenizer` argument.
  - Root cause: Passing a tokenizer instance instead of a string path.
  - Fix: Only pass the directory path to vLLM (`tokenizer=str(ckpt_dir)`). Status: fixed in `from_local()`.

- __EnTokenizer.__init__() missing required `vocab_file`__
  - Symptom: `TypeError: EnTokenizer.__init__() missing 1 required positional argument: 'vocab_file'` from site-packages path.
  - Root cause: Importing the installed package version whose old code tried `EnTokenizer()` directly.
  - Fix: Ensure local `src/` wins in `sys.path` or pin the correct Git revision. Status: mitigated by `sys.path.insert(0, src)`.

- __AutoTokenizer slow/fast both fail with__ `ValueError: not enough values to unpack (expected 2, got 1)`
  - Symptom: Both `use_fast=False` and `use_fast=True` `AutoTokenizer.from_pretrained(...)` fail.
  - Root cause: Transformers received a bare class name during dynamic import (e.g., `"EnTokenizer"`), typically due to `tokenizer_class` or misconfigured mapping.
  - Fix: Remove `tokenizer_class` from all configs. Use `auto_map.AutoTokenizer` list form plus `tokenizer_file` and `use_fast: true`. Status: updated; awaiting Modal logs to confirm resolution.

- __Fast path unresolved__ (pre-change)
  - Symptom: `use_fast=True` did not resolve to `tokenizer.EnTokenizer` even with `auto_map`.
  - Root cause: `auto_map.AutoTokenizer` was a single string; some TF/vLLM paths prefer list form `[slow, fast]`.
  - Fix: Set both entries to `"tokenizer.EnTokenizer"`. Status: updated; awaiting Modal logs.

- __Model auto_map error: `ValueError: too many values to unpack (expected 2)`__
  - Symptom: During vLLM/Transformers dynamic loading of model class from `config.json`, an unpacking error occurs.
  - Root cause: Using an overly deep module path for `AutoModel` in `auto_map` confused the dynamic module resolver.
  - Fix: Write a `t3.py` shim into the model dir and set `auto_map.AutoModel` to `"t3.ChatterboxT3"`. Also set env var `VLLM_MODEL_IMPLEMENTATION=t3.ChatterboxT3`. Status: fixed.

## Expected logs (happy path)
- "MODEL_DIR: /root/t3-model"
- Downloads for safetensors and `tokenizer.json`
- "tokenizer_config.json: { ... auto_map, tokenizer_file, use_fast ... }"
- Preflight tokenizer resolution (from `app.py`):
  - `AutoTokenizer slow class: tokenizer.EnTokenizer`
  - `AutoTokenizer fast class: tokenizer.EnTokenizer`
  - `EnTokenizer.from_pretrained signature: (pretrained_model_name_or_path, *inputs, **kwargs)`
- "Loading tokenizer from: t3-model/tokenizer.json" and `EnTokenizer(vocab_file=...)` test encode succeeds
- `chatterbox_vllm.tts` logs:
  - "Initialized T3Config"
  - "Loaded t3_cfg.safetensors"
  - "Loaded and initialized T3CondEnc / t3_speech_emb / t3_speech_pos_emb"
  - "Calculated vLLM memory: ..."
  - "Using model_path=/root/t3-model, tokenizer_path=/root/t3-model"
  - vLLM args include `"tokenizer": "/root/t3-model"` and `"tokenizer_mode": "fast"` (or `"auto"`)
  - "Initialized vLLM LLM"
- First request will be slow due to downloads and GPU init

## Modal image tips
- Prefer importing local `src/` to avoid mismatch with installed package. The order of `sys.path` decides.
- If you want to install from GitHub instead, pin with `@branch-or-sha` and remove local `src/` to avoid ambiguity.
- GPU memory utilization is computed dynamically in `from_local()`; you can tune via `max_batch_size`, `max_model_len`, or override `gpu_memory_utilization` in args.

## Diagnostics to capture on every deploy
- __Log library versions__ (early in `app.py`):
  ```python
  import transformers, tokenizers, vllm
  log.info(f"versions: transformers={transformers.__version__}, tokenizers={tokenizers.__version__}, vllm={getattr(vllm, '__version__', 'unknown')}")
  ```
- __Dump configs__ (already present): contents of `t3-model/config.json` and `t3-model/tokenizer_config.json`.
- __Preflight tokenizer resolution__ (already present): log resolved classes for `use_fast=False/True` with `local_files_only=True`.

## Switch to fast path (after slow preflight passes)
- Keep `EnTokenizer` in `tokenizer.py` and the HF configs as described above.
- Confirm in logs that `AutoTokenizer(use_fast=True)` resolves to `tokenizer.EnTokenizer`.
- Set `tokenizer_mode` to `"fast"` (or drop it to use `"auto"`) in the call to `ChatterboxTTS.from_local()`.

## Troubleshooting
- __ValueError: not enough values to unpack (expected 2, got 1)__
  - Meaning: Transformers received a bare tokenizer class name (e.g., `"EnTokenizer"`) instead of a dotted path, while trying dynamic import.
  - Fix: Remove `tokenizer_class` from both configs, rely on `auto_map` list + `tokenizer_file` + `use_fast: true`.
  - Ensure `tokenizer.py` exists in the model dir and `trust_remote_code: true` is set.
- __AutoTokenizer fast fails but slow succeeds__
  - Ensure `auto_map.AutoTokenizer` is a list `[slow, fast]` and both entries are `"tokenizer.EnTokenizer"`.
  - Verify `tokenizer_file: "tokenizer.json"` is present in both configs.
  - Keep `local_files_only=True` during preflight to avoid remote cache effects.

- __Still seeing long `AutoModel` path in `config.json` logs__
  - Meaning: The running container is using an older build of `app.py` where `auto_map.AutoModel` is `"chatterbox_vllm.models.t3.modules.t3.ChatterboxT3"`.
  - Check logs for `Wrote t3.py shim ...`. If missing, you're on a stale image.
  - Fix: Force rebuild/redeploy so the latest `app.py` is used. After redeploy, logs should show:
    - `Wrote t3.py shim to t3-model/t3.py`
    - `config.json` printed with `"AutoModel": "t3.ChatterboxT3"`.

## Current status (to be validated on next deploy)
- Configs now use `auto_map.AutoTokenizer` list form, include `tokenizer_file` and `use_fast: true`, and omit `tokenizer_class`.
- Preflight loads in `app.py` use `local_files_only=True` and log resolved classes for slow/fast.
- Action: Redeploy and verify logs show `tokenizer.EnTokenizer` for both slow/fast. Then set `tokenizer_mode` to `"fast"`.


## Quick checklist
- __Imports use local code__: `log __file__` of `chatterbox_vllm` points to `.../src/chatterbox_vllm/__init__.py`.
- __Tokenizer files exist__: `t3-model/tokenizer.json` and `t3-model/tokenizer.py`.
- __Configs correct (no tokenizer_class)__: both `config.json` and `tokenizer_config.json` have `auto_map.AutoTokenizer` list set to `"tokenizer.EnTokenizer"`, contain `tokenizer_file` and `use_fast: true`, and set `trust_remote_code: true`.
- __No tokenizer object passed to vLLM__: only pass the directory path.
- __Custom model wired__: `VLLM_MODEL_IMPLEMENTATION=t3.ChatterboxT3` + `model_loader_extra_config`.
- __t3.py shim present__: `t3-model/t3.py` exists and re-exports `ChatterboxT3`.
- __First call warm-up__: expect tens of seconds on cold start.
