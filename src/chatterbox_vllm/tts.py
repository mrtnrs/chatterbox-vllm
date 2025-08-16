from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import time
import os
from vllm import LLM, SamplingParams
from functools import lru_cache
import librosa
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from chatterbox_vllm.models.t3.modules.t3_config import T3Config
from chatterbox_vllm.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox_vllm.models.s3gen import S3GEN_SR, S3Gen
from chatterbox_vllm.models.voice_encoder import VoiceEncoder
from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond, T3CondEnc
from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.text_utils import punc_norm

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dictdef to(self, device):
    self.t3 = self.t3.to(device=device)
    for k, v in self.gen.items():
        if torch.is_tensor(v):
            self.gen[k] = v.to(device=device)
    return self

@classmethod
def load(cls, fpath):
    kwargs = torch.load(fpath, weights_only=True)
    return cls(T3Cond(**kwargs['t3']), kwargs['gen'])class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SRdef __init__(self, target_device: str, max_model_len: int,
             t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
             t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
             s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals):
    self.target_device = target_device
    self.max_model_len = max_model_len
    self.t3 = t3
    self.t3_config = t3_config
    self.t3_cond_enc = t3_cond_enc
    self.t3_speech_emb = t3_speech_emb
    self.t3_speech_pos_emb = t3_speech_pos_emb
    self.s3gen = s3gen
    self.ve = ve
    self.default_conds = default_conds

@property
def sr(self) -> int:
    """Sample rate of synthesized audio"""
    return S3GEN_SR

@classmethod
def from_local(cls, ckpt_dir: str, target_device: str = "cuda", 
               max_model_len: int = 1000, compile: bool = False,
               max_batch_size: int = 10,
               **kwargs) -> 'ChatterboxTTS':
    import logging
    log = logging.getLogger(__name__)
    log.info(f"Starting ChatterboxTTS.from_local with ckpt_dir={ckpt_dir}, target_device={target_device}, max_model_len={max_model_len}")
    log.info(f"Received kwargs: {kwargs}")

    ckpt_dir = Path(ckpt_dir)
    log.info(f"Resolved ckpt_dir to: {ckpt_dir.absolute()}")

    t3_config = T3Config()
    log.info("Initialized T3Config")

    # Load necessary weights for T3CondEnc
    log.info(f"Loading weights from {ckpt_dir / 't3_cfg.safetensors'}")
    t3_weights = load_file(ckpt_dir / "t3_cfg.safetensors")
    log.info("Loaded t3_cfg.safetensors")

    t3_enc = T3CondEnc(t3_config)
    t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
    t3_enc = t3_enc.to(device=target_device).eval()
    log.info("Loaded and initialized T3CondEnc")

    t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
    t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
    t3_speech_emb = t3_speech_emb.to(device=target_device).eval()
    log.info("Loaded and initialized t3_speech_emb")

    t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
    t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
    t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()
    log.info("Loaded and initialized t3_speech_pos_emb")

    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
    vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
    vllm_memory_percent = vllm_memory_needed / unused_gpu_memory
    log.info(f"Calculated vLLM memory: {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

    model_path = os.path.abspath(str(ckpt_dir))
    # Use tokenizer path from kwargs if provided, else fall back to ckpt_dir
    tokenizer_path = os.path.abspath(str(kwargs.pop('tokenizer', ckpt_dir)))
    log.info(f"Using model_path={model_path}, tokenizer_path={tokenizer_path}")
    log.info(f"Remaining kwargs after pop: {kwargs}")

    # Verify tokenizer files exist
    tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer_py_path = os.path.join(tokenizer_path, "tokenizer.py")
    log.info(f"Verifying tokenizer files: {tokenizer_json_path}, {tokenizer_py_path}")
    if not os.path.exists(tokenizer_json_path) or not os.path.exists(tokenizer_py_path):
        log.error(f"Tokenizer files missing: {tokenizer_json_path}, {tokenizer_py_path}")
        raise FileNotFoundError(f"Tokenizer files missing: {tokenizer_json_path}, {tokenizer_py_path}")
    # Log tokenizer.json contents
    try:
        with open(tokenizer_json_path, 'r') as f:
            log.info(f"tokenizer.json contents (first 100 chars): {f.read()[:100]}...")
    except Exception as e:
        log.error(f"Failed to read tokenizer.json: {e}", exc_info=True)
        raise

    # Prepare LLM arguments
    llm_kwargs = {
        "model": model_path,
        "task": "generate",
        "tokenizer": tokenizer_path,
        "tokenizer_mode": "auto",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": vllm_memory_percent,
        "enforce_eager": not compile,
        "trust_remote_code": True,
    }
    llm_kwargs.update(kwargs)  # Add remaining kwargs
    log.info(f"Calling vLLM LLM with arguments: {llm_kwargs}")

    t3 = LLM(**llm_kwargs)
    log.info("Initialized vLLM LLM")

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve = ve.to(device=target_device).eval()
    log.info("Loaded and initialized VoiceEncoder")

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen = s3gen.to(device=target_device).eval()
    log.info("Loaded and initialized S3Gen")

    default_conds = Conditionals.load(ckpt_dir / "conds.pt")
    default_conds.to(device=target_device)
    log.info("Loaded and initialized default conditionals")

    instance = cls(
        target_device=target_device, max_model_len=max_model_len,
        t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
        s3gen=s3gen, ve=ve, default_conds=default_conds,
    )
    log.info("Created ChatterboxTTS instance")
    return instance

@classmethod
def from_pretrained(cls, ckpt_dir: str = "./t3-model", *args, **kwargs) -> 'ChatterboxTTS':
    # Assume ckpt_dir already contains all necessary files
    return cls.from_local(Path(ckpt_dir), *args, **kwargs)

@lru_cache(maxsize=10)
def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
    if wav_fpath is None:
        s3gen_ref_dict = self.default_conds.gen
        t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
        ve_embed = self.default_conds.t3.speaker_emb
    else:
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)
        s3_tokzr = self.s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True)

    cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)
    cond_emb = self.t3_cond_enc(T3Cond(
        speaker_emb=ve_embed,
        cond_prompt_speech_tokens=t3_cond_prompt_tokens,
        cond_prompt_speech_emb=cond_prompt_speech_emb,
        emotion_adv=0.5 * torch.ones(1, 1)
    ).to(device=self.target_device)).to(device="cpu")

    return s3gen_ref_dict, cond_emb

def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
    if exaggeration == 0.5:
        return cond_emb
    new_cond_emb = cond_emb.clone()
    new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
        (exaggeration * torch.ones(1, 1)).to(self.target_device)
    ).to('cpu')
    return new_cond_emb

def generate(
    self,
    prompts: Union[str, list[str]] = None,
    prompt_token_ids: Optional[list[list[int]]] = None,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    max_tokens: int = 1000,
    top_p: float = 0.8,
    repetition_penalty: float = 2.0,
    *args, **kwargs,
) -> list[any]:
    s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)
    return self.generate_with_conds(
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        s3gen_ref=s3gen_ref,
        cond_emb=cond_emb,
        temperature=temperature,
        exaggeration=exaggeration,
        max_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        *args, **kwargs
    )

def generate_with_conds(
    self,
    prompts: Union[str, list[str]] = None,
    prompt_token_ids: Optional[list[list[int]]] = None,
    s3gen_ref: dict[str, Any] = None,
    cond_emb: torch.Tensor = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    max_tokens: int = 1000,
    top_p: float = 0.8,
    repetition_penalty: float = 2.0,
    *args, **kwargs,
) -> list[any]:
    if prompts is None and prompt_token_ids is None:
        raise ValueError("Either prompts or prompt_token_ids must be provided.")
    if prompts is not None and prompt_token_ids is not None:
        raise ValueError("Cannot provide both prompts and prompt_token_ids.")

    cond_emb = self.update_exaggeration(cond_emb, exaggeration)

    if prompts is not None:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]
        prompt_token_ids_list = [None] * len(prompts)
    else:
        prompts = [None] * len(prompt_token_ids)
        prompt_token_ids_list = prompt_token_ids

    with torch.inference_mode():
        start_time = time.time()
        batch_results = self.t3.generate(
            [
                {
                    "prompt": prompt,
                    "prompt_token_ids": token_ids,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                }
                for prompt, token_ids in zip(prompts, prompt_token_ids_list)
            ],
            sampling_params=SamplingParams(
                temperature=temperature,
                stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                max_tokens=min(max_tokens, self.max_model_len),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                *args, **kwargs,
            )
        )
        t3_gen_time = time.time() - start_time
        print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

        torch.cuda.empty_cache()

        start_time = time.time()
        results = []
        for i, batch_result in enumerate(batch_results):
            for output in batch_result.outputs:
                if i % 5 == 0:
                    print(f"[S3] Processing prompt {i} of {len(batch_results)}")
                if i % 10 == 0:
                    torch.cuda.empty_cache()

                speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561]

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref,
                )
                results.append(wav.cpu())
        s3gen_gen_time = time.time() - start_time
        print(f"[S3Gen] Waveform Generation time: {s3gen_gen_time:.2f}s")

        return results

def shutdown(self):
    del self.t3
    torch.cuda.empty_cache()

