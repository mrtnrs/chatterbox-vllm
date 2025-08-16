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

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
HARDCODED_API_KEY = "ilovefrietenmetmayo"

log.info(f"Python version: {sys.version}")
log.info(f"Running in Modal: {os.getenv('MODAL_ENVIRONMENT', 'local')}")

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
        f"git+https://github.com/mrtnrs/chatterbox-vllm",
        "pydantic>=2.0,<3.0",
        "hf_transfer>=0.1.6",
        "tokenizers>=0.19.1",
        "transformers>=4.30.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/root/.cache/huggingface",
    })
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_MODEL_IMPLEMENTATION"] = "chatterbox_vllm.models.t3.modules.t3.ChatterboxT3"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("chatterbox-vllm-tts")

# -------------------- Model Loader --------------------
_model = None
_sr = 24000
MODEL_DIR = "t3-model"

HARDCODED_CONFIG_JSON = {
    "architectures": ["ChatterboxT3"],
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
        "AutoModel": "chatterbox_vllm.models.t3.modules.t3.ChatterboxT3"
    }
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

    # -------------------- Ensure Model Dir --------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info(f"MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    
    files_to_download = ["t3_cfg.safetensors", "ve.safetensors", 
                         "s3gen.safetensors", "conds.pt", "tokenizer.json"]
    for fpath in files_to_download:
        log.info(f"Downloading {fpath}...")
        hf_hub_download("ResembleAI/chatterbox", fpath, local_dir=MODEL_DIR)

    import chatterbox_vllm
    shutil.copy(
        os.path.join(chatterbox_vllm.__path__[0], "models", "t3", "entokenizer.py"),
        os.path.join(MODEL_DIR, "tokenizer.py"),
    )

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(HARDCODED_CONFIG_JSON, f, indent=2)

    # Write tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "EnTokenizer",
        "trust_remote_code": True
    }
    with open(os.path.join(MODEL_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Proper tokenizer initialization
    sys.path.append(MODEL_DIR)
    from tokenizer import EnTokenizer
    
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
        tokenizer=MODEL_DIR
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

        log.info("Generating speech...")
        audios = model.generate(
            [payload.text],
            audio_prompt_path=audio_prompt_path,
            exaggeration=payload.exaggeration
        )

        buf = io.BytesIO()
        ta.save(buf, audios[0], model.sr, format="mp3")
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'}
        )

    return web_app
