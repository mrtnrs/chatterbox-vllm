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