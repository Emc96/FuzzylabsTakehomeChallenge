import os

# facebook/nllb-200-distilled-600M is a single multilingual model covering 200
# languages, chosen over per-pair models (e.g. Helsinki-NLP/opus-mt-*) so
# that adding a new language pair is a config change, not a new model to
# load and manage. See README for the full rationale.
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/nllb-200-distilled-600M")


NLLB_LANG_MAP = {
    "en": "eng_Latn", 
    "fr": "fra_Latn", 
    "de": "deu_Latn", 
    "es": "spa_Latn", 
    "it": "ita_Latn", 
    "pt": "por_Latn", 
    "nl": "nld_Latn",
    "ru": "rus_Cyrl", 
    "zh": "zho_Hans", 
    "ja": "jpn_Jpan", 
    "ar": "arb_Arab", 
    "uk": "ukr_Cyrl"
}

MAX_INPUT_LENGTH = 5000

MAX_BATCH_SZIE = 10
# this is trading latency for throughput, higher means getting more requests per batch
# but more delay per request
MAX_WAIT_MS = 50