import io
import os
import json
import logging
from typing import Optional
import modal
from fastapi import HTTPException, Response, Header
import shutil
import sys
import time
import inspect
import importlib
# Heavy deps are optional locally; import lazily later
transformers = None  # type: ignore
tokenizers = None  # type: ignore
_vllm = None  # type: ignore
torch = None  # type: ignore

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
HARDCODED_API_KEY = "ilovefrietenmetmayo"

log.info(f"Python version: {sys.version}")
log.info(f"Running in Modal: {os.getenv('MODAL_ENVIRONMENT', 'local')}")

# Ensure local src/ is importable before any package fallback
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path and os.path.isdir(SRC_DIR):
    sys.path.insert(0, SRC_DIR)

# -------------------- Modal Image --------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "lame", "git")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "fastapi[standard]>=0.115",
        "requests>=2.31",
        "huggingface_hub>=0.25.1",
        "safetensors>=0.3.0",
        "vllm==0.10.0",
        f"git+https://github.com/mrtnrs/chatterbox-vllm#343553",
        "pydantic>=2.0,<3.0",
        "hf_transfer>=0.1.6",
        "tokenizers>=0.19.1",
        "transformers>=4.30.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "VLLM_NO_USAGE_STATS": "1",
    })
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_MODEL_IMPLEMENTATION"] = "t3.ChatterboxT3"
try:
    import transformers as transformers  # noqa: F401
    import tokenizers as tokenizers  # noqa: F401
    import vllm as _vllm  # noqa: F401
    import torch as torch  # noqa: F401
    log.info(
        f"versions: transformers={transformers.__version__}, "
        f"tokenizers={tokenizers.__version__}, vllm={getattr(_vllm, '__version__', 'unknown')}, torch={torch.__version__}"
    )
    log.info(f"CUDA available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
except Exception as e:
    log.warning(f"Optional deps not available locally: {e}. Will import inside _get_model() when running in container.")
log.info(f"VLLM_MODEL_IMPLEMENTATION={os.environ.get('VLLM_MODEL_IMPLEMENTATION')}")
log.info(f"sys.path[0:5]={sys.path[:5]}")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("chatterbox-vllm-tts")

# -------------------- Model Loader --------------------
_model = None
_sr = 24000
MODEL_DIR = "t3-model"

HARDCODED_CONFIG_JSON = {
    "architectures": "t3.ChatterboxT3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_implementation": "sdpa",
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 30,
    "num_key_value_heads": 16,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 8,
    "auto_map": {  # CRITICAL ADDITION
        "AutoModel": "t3.ChatterboxT3"
    },
    "tokenizer_file": "tokenizer.json",
    "use_fast": True,
}

def _get_model():
    global _model, _sr
    if _model is not None:
        log.info("Model already loaded. Re-using cached model.")
        return _model

    log.info("Importing chatterbox_vllm modules...")
    from chatterbox_vllm.tts import ChatterboxTTS
    from chatterbox_vllm.text_utils import punc_norm
    from huggingface_hub import hf_hub_download
    import chatterbox_vllm as _cb
    log.info(f"Using chatterbox_vllm from: {_cb.__file__}")
    # Import heavy deps inside container/runtime
    try:
        import transformers as transformers  # noqa: F401
        import tokenizers as tokenizers  # noqa: F401
        import vllm as _vllm  # noqa: F401
        import torch as torch  # noqa: F401
        log.info(
            f"versions (container): transformers={transformers.__version__}, "
            f"tokenizers={tokenizers.__version__}, vllm={getattr(_vllm, '__version__', 'unknown')}, torch={torch.__version__}"
        )
        log.info(f"CUDA available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    except Exception as e:
        log.warning(f"Heavy deps import in _get_model failed: {e}")

    # -------------------- Ensure Model Dir --------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info(f"MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    
    files_to_download = ["t3_cfg.safetensors", "ve.safetensors", 
                         "s3gen.safetensors", "conds.pt", "tokenizer.json"]
    for fpath in files_to_download:
        log.info(f"Downloading {fpath}...")
        hf_hub_download("ResembleAI/chatterbox", fpath, local_dir=MODEL_DIR)
    # Verify downloaded files
    for fpath in files_to_download:
        full = os.path.join(MODEL_DIR, fpath)
        exists = os.path.exists(full)
        size = os.path.getsize(full) if exists else -1
        log.info(f"Verified file: {full} exists={exists} size={size}")

    # Write a fresh tokenizer.py to MODEL_DIR to ensure correct from_pretrained signature
    dest_tok = os.path.join(MODEL_DIR, "tokenizer.py")
    TOKENIZER_PY = '''
import logging
import os
from typing import List, Optional, Union

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer(PreTrainedTokenizerFast):
    """
    A VLLM-compatible fast tokenizer that wraps the rust-based Tokenizer.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        tokenizer_object = Tokenizer.from_file(vocab_file)
        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = text.replace(' ', SPACE)
        return super()._tokenize(text, **kwargs)

    def encode(self, txt: str, verbose=False, return_tensors: Optional[str] = None, add_special_tokens: bool = True, **kwargs):
        """Override for custom preprocessing; supports legacy params."""
        txt = txt.replace(' ', SPACE)
        encoded = super().encode(txt, add_special_tokens=add_special_tokens, **kwargs)
        if return_tensors == "pt":
            return torch.tensor(encoded).unsqueeze(0)
        return encoded

    def decode(self, seq, **kwargs):
        """Override for custom postprocessing; supports legacy params."""
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt: str = super().decode(seq, **kwargs)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save the tokenizer to a directory.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self._tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

    def text_to_tokens(self, text: str):
        """Legacy method for backward compatibility"""
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    @property
    def max_token_id(self) -> int:
        return max(self.get_vocab().values())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a tokenizer from a pretrained model or path.
    
        Args:
            pretrained_model_name_or_path: Path to the directory containing tokenizer.json
            *inputs: Additional positional arguments
            **kwargs: Additional keyword arguments to pass to the tokenizer
        """
        vocab_file = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
        if not os.path.exists(vocab_file):
            raise ValueError(f"tokenizer.json not found at {pretrained_model_name_or_path}")
        return cls(vocab_file=vocab_file, *inputs, **kwargs)
'''
    with open(dest_tok, "w") as f:
        f.write(TOKENIZER_PY)
    log.info(f"Wrote tokenizer.py to {dest_tok}")

    # Write a lightweight t3.py shim so AutoModel can import t3.ChatterboxT3
    t3_py_path = os.path.join(MODEL_DIR, "t3.py")
    T3_PY = '''
"""
Minimal t3.py shim for Transformers/vLLM dynamic import.

Purpose:
- Allow dynamic_module_utils.get_class_from_dynamic_module to import
  a class named `ChatterboxT3` from the module `t3` referenced by
  config.json's auto_map ("AutoModel": "t3.ChatterboxT3").

Notes:
- This class is a minimal stub to satisfy architecture resolution.
- vLLM will not use this class for execution; it loads weights via its
  own runtime. If instantiated by Transformers, this class simply
  creates a bare PreTrainedModel which should be sufficient for any
  metadata checks.
"""

from transformers import PreTrainedModel, PretrainedConfig


class ChatterboxT3(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(config)


__all__ = ["ChatterboxT3"]
'''
    with open(t3_py_path, "w") as f:
        f.write(T3_PY)
    log.info(f"Wrote t3.py shim to {t3_py_path}")
    # Ensure MODEL_DIR is importable and test shim import
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    try:
        mod = importlib.import_module("t3")
        cls = getattr(mod, "ChatterboxT3", None)
        log.info(
            f"Shim import test: module={mod.__name__} file={getattr(mod, '__file__', '?')} class_ok={cls is not None}"
        )
    except Exception as e:
        log.exception(f"Shim import test failed: {e}")

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(HARDCODED_CONFIG_JSON, f, indent=2)
    # Log model config.json for diagnostics
    try:
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            log.info(f"config.json: {f.read()}")
    except Exception as e:
        log.exception(f"Failed reading config.json: {e}")
    log.info(f"auto_map in HARDCODED_CONFIG_JSON: {HARDCODED_CONFIG_JSON.get('auto_map')}")

    # Write tokenizer_config.json (favor auto_map; avoid bare tokenizer_class)
    tokenizer_config = {
        "auto_map": {
            "AutoTokenizer": ["tokenizer.EnTokenizer", "tokenizer.EnTokenizer"]
        },
        "tokenizer_file": "tokenizer.json",
        "use_fast": True,
        "trust_remote_code": True
    }
    with open(os.path.join(MODEL_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(os.path.join(MODEL_DIR, "tokenizer_config.json")) as f:
        log.info(f"tokenizer_config.json: {f.read()}")
    # Preflight AutoTokenizer loads to mirror vLLM behavior
    try:
        from transformers import AutoTokenizer
        tok_slow = AutoTokenizer.from_pretrained(os.path.abspath(MODEL_DIR), trust_remote_code=True, use_fast=False, local_files_only=True)
        log.info(f"AutoTokenizer slow class: {tok_slow.__class__.__module__}.{tok_slow.__class__.__name__}")
    except Exception as e:
        log.exception(f"AutoTokenizer slow failed: {e}")
    try:
        from transformers import AutoTokenizer as _AT2
        tok_fast = _AT2.from_pretrained(os.path.abspath(MODEL_DIR), trust_remote_code=True, use_fast=True, local_files_only=True)
        log.info(f"AutoTokenizer fast class: {tok_fast.__class__.__module__}.{tok_fast.__class__.__name__}")
    except Exception as e:
        log.exception(f"AutoTokenizer fast failed: {e}")
    
    # Proper tokenizer initialization
    sys.path.append(MODEL_DIR)
    from tokenizer import EnTokenizer
    # Verify runtime signature to avoid mismatch
    log.info(f"EnTokenizer.from_pretrained signature: {inspect.signature(EnTokenizer.from_pretrained)}")
    
    # Get the path to tokenizer.json
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    log.info(f"Loading tokenizer from: {tokenizer_path}")
    
    # Initialize tokenizer correctly
    tok = EnTokenizer(vocab_file=tokenizer_path)
    test_text = "[START]Hello world[STOP]"
    log.info(f"Tokenizer test: '{test_text}' → {tok.encode(test_text)}")

    # -------------------- Load model --------------------
    log.info("Loading ChatterboxTTS model...")
    _model = ChatterboxTTS.from_local(
        ckpt_dir=MODEL_DIR,
        tokenizer=os.path.abspath(MODEL_DIR),
        tokenizer_mode="auto",
        model_loader_extra_config={"model_class": "t3.ChatterboxT3"}
    )
    _sr = _model.sr
    log.info(f"Model loaded successfully (SR={_sr})")

    return _model

# def _get_model():
#     global _model, _sr
#     if _model is not None:
#         log.info("Model already loaded. Re-using cached model.")
#         return _model

#     # os.environ["VLLM_SKIP_MODEL_LOAD"] = "1"
#     log.info("Importing chatterbox_vllm modules...")
#     try:
#         from chatterbox_vllm.tts import ChatterboxTTS
#         from chatterbox_vllm.text_utils import punc_norm
#     except Exception as e:
#         log.error(f"Failed to import chatterbox_vllm modules: {e}", exc_info=True)
#         raise

#     from huggingface_hub import hf_hub_download

#     # -------------------- Ensure Model Dir --------------------
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     log.info(f"MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    
#     # Step 1: Download model files
#     files_to_download = ["t3_cfg.safetensors", "ve.safetensors", "s3gen.safetensors", "conds.pt", "tokenizer.json"]
#     for fpath in files_to_download:
#         log.info(f"Downloading {fpath}...")
#         local_path = hf_hub_download("ResembleAI/chatterbox", fpath, local_dir=MODEL_DIR)
#         log.info(f"Downloaded {fpath} → {local_path}")

#     # Step 2: Copy entokenizer.py
#     import chatterbox_vllm
#     entokenizer_src = os.path.join(chatterbox_vllm.__path__[0], 'models', 't3', 'entokenizer.py')
#     tokenizer_dest = os.path.join(MODEL_DIR, 'tokenizer.py')
#     shutil.copy(entokenizer_src, tokenizer_dest)
#     log.info(f"Copied entokenizer.py to {tokenizer_dest}")

#     # Step 2b: Write config.json
#     config_dest = os.path.join(MODEL_DIR, "config.json")
#     with open(config_dest, "w") as f:
#         json.dump(HARDCODED_CONFIG_JSON, f, indent=2)
#     log.info(f"Wrote config.json → {config_dest}")

#     # Step 3: Tokenizer sanity check
#     sys.path.append(MODEL_DIR)
#     try:
#         from tokenizer import EnTokenizer
#         tok = EnTokenizer.from_pretrained()
#         test_text = "[START]Hello world[STOP]"
#         test_ids = tok.encode(test_text)
#         log.info(f"Tokenizer test: '{test_text}' → {test_ids}")
#     except Exception as e:
#         log.error(f"Tokenizer sanity check failed: {e}", exc_info=True)
#         raise

#     # Step 4: Load ChatterboxTTS
#     log.info("Loading ChatterboxTTS model...")
#     _model = ChatterboxTTS.from_local(
#         ckpt_dir=MODEL_DIR,
#         use_vllm=False
#     )
#     _sr = _model.sr
#     log.info(f"Model loaded successfully (SR={_sr})")

#     return _model

# -------------------- ASGI App --------------------
@app.function(
    image=vllm_image,
    gpu="A10G",
    timeout=600,
    min_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.asgi_app()
def generate_tts():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import requests
    import torchaudio as ta

    web_app = FastAPI()

    class GenerateRequest(BaseModel):
        text: str
        audio_prompt_url: Optional[str] = None
        exaggeration: float = 0.8

    @web_app.post("/")
    async def tts_endpoint(payload: GenerateRequest, x_api_key: str = Header(..., alias="X-API-Key")):
        if x_api_key != HARDCODED_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        log.info(f"Received text: '{payload.text[:100]}...'")
        model = _get_model()

        audio_prompt_path = None
        if payload.audio_prompt_url:
            r = requests.get(payload.audio_prompt_url, timeout=20)
            r.raise_for_status()
            audio_prompt_path = "/tmp/ref_audio.mp3"
            with open(audio_prompt_path, "wb") as f:
                f.write(r.content)
            try:
                _sz = os.path.getsize(audio_prompt_path)
            except Exception:
                _sz = -1
            log.info(f"Downloaded audio prompt → {audio_prompt_path} ({_sz} bytes)")

        log.info("Generating speech...")
        audios = model.generate(
            [payload.text],
            audio_prompt_path=audio_prompt_path,
            exaggeration=payload.exaggeration
        )
        try:
            log.info(f"Generated audio shape={audios[0].shape}, dtype={audios[0].dtype}, sr={model.sr}")
        except Exception:
            pass

        buf = io.BytesIO()
        ta.save(buf, audios[0], model.sr, format="mp3")
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'}
        )

    return web_app
