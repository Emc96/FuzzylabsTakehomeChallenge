"""
This script is a small wrapper around the translaiton model, NLLB-200.

Model & tokeniser is loaded once rather than per request, reduce the latency.
This loading is done seperately from the actual translation of a request & in the load_model function
so that testing etc can be done without having to load the whole model, better for unit testing.
"""

import logging

from .config import MODEL_NAME, NLLB_LANG_MAP

logger = logging.get(__name__)

tokeniser = None
model = None


def load_model():
    """
    Downloads the model on the first run, loads the weights on subsequent runs
    Returns:
        tokensier Autotokenizer: Huggingface transformer autotokenizer object for the model
        model: AutoModelForSeq2SeqLM: Huggingface
    """

    global tokeniser, model

    # check if model is none,
    if model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logging.info("Loading model: %s", MODEL_NAME)

        tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        # disable training mode, model should return results faster
        model.eval()
        logging.info("Model loaded")

    return tokeniser, model


def warm_up():
    """
    Function is used on server start up to load model before taking
    requests
    """
    load_model()


def translate_batch(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """
    This will be passed a list of strings that share the same source and target language pair.
    The source language and the target language. It uses this information to tokenise inputs and generate
    the texts into the correct language for translation.
    Returning a batch of translated text.

    Args:
        texts (list[str]): A batch of strings that need translation
        src_lang (str): source language code sent to the API
        tgt_lang (str): target language code sent to the API

    Returns:
        list[str]: The translated texts stored in a list for access in the API
    """
    # get the already loaded/warmed_up tokeniser and model objects
    tokeniser, model = load_model()

    source_code = NLLB_LANG_MAP.get(src_lang)
    target_code = NLLB_LANG_MAP.get(tgt_lang)

    tokeniser.src_lang = source_code
    # since requests can be of a different length, need to pad short sequences with 0s
    tokens = tokeniser(texts, return_tensors="pt", padding=True, truncation=True)

    # forced_bos_token_id makes the model output into the target language
    generated_repsonse = model.generate(
        **tokens,
        forced_bos_token_id=tokeniser.convert_tokens_to_ids(target_code),
    )
    # batch decode turns the model output into human readable text, removing the AI BOS/EOS tokens
    return tokeniser.batch_decode(generated_repsonse, skip_special_tokens=True)
