"""
Translation module powered by AI4Bharat IndicTrans2 model.
Supports translation between English and supported Indic languages.
"""
import os
from typing import List, Union

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from IndicTransToolkit.IndicTransToolkit import IndicProcessor

# Load environment and authenticate with HuggingFace
load_dotenv(override=True)
_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)

_iproc = IndicProcessor(inference=True)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Reverse model configuration for English->Indic (requires gated access)
_REV_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
_rev_tokenizer = None
_rev_model = None

# ISO code to FLORES code mapping for IndicProcessor
ISO2FLORES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "or": "ory_Orya",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
}

def _load_reverse_model():
    """
    Lazy-load the English->Indic reverse translation model.
    """
    global _rev_tokenizer, _rev_model
    if _rev_model is None:
        _rev_tokenizer = AutoTokenizer.from_pretrained(
            _REV_MODEL_NAME, trust_remote_code=True
        )
        _rev_model = AutoModelForSeq2SeqLM.from_pretrained(
            _REV_MODEL_NAME, trust_remote_code=True
        )
        _rev_model.to(_DEVICE)

_load_reverse_model()

def translate(
    texts: Union[str, List[str]],
    target_lang: str,
    src_lang: str = "en",
) -> Union[str, List[str]]:
    """
    Translate text(s) from src_lang to target_lang using IndicTrans2.
    """
    # No-op if languages are identical
    if src_lang == target_lang:
        return texts

    # Validate language codes
    try:
        src_flores = ISO2FLORES[src_lang]
        tgt_flores = ISO2FLORES[target_lang]
    except KeyError:
        supported = sorted(ISO2FLORES.keys())
        raise ValueError(f"Unsupported language; supported codes: {supported}")

    # Determine translation direction and select model/tokenizer
    if src_flores == ISO2FLORES['en'] and tgt_flores != ISO2FLORES['en']:
        if _rev_tokenizer is None or _rev_model is None:
            raise RuntimeError("Reverse translation model is not available")
        tokenizer = _rev_tokenizer  # type: ignore[assignment]
        model = _rev_model  # type: ignore[assignment]
    else:
        raise NotImplementedError(
            f"Translation from {src_lang} to {target_lang} is not supported"
        )

    # Prepare batch
    is_single = isinstance(texts, str)
    # Prepare batch
    is_single = isinstance(texts, str)
    batch_texts = [texts] if is_single else list(texts)
    inputs = _iproc.preprocess_batch(batch_texts, src_lang=src_flores, tgt_lang=tgt_flores)

    # Tokenize
    encodings = tokenizer(
        inputs,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(_DEVICE)

    # Generate translations
    with torch.no_grad():
        generated = model.generate(
            **encodings,
            use_cache=False,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode
    with tokenizer.as_target_tokenizer():  # type: ignore[attr-defined]
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess placeholders, etc.
    outputs = _iproc.postprocess_batch(decoded, lang=tgt_flores)
    return outputs[0] if is_single else outputs