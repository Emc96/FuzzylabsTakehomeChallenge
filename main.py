import logging 
import time 

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request 
from pydantic import BaseModel, Field 
from .application.batching import BatchingTranslator 
from .application import translation
from .application.config import NLLB_LANG_MAP

logging.basicConfig(
    format="%(asctime)s -  %(levelname)s : %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="TranslationLLM.log",
)


class TranslationRequest(BaseModel):
    # the ellipses states this is an required field but does not have a default value
    # fastapi will reject the request if user does not supply these fields
    text: str = Field(..., min_length=1, max_length=5000, examples=["Good morning, weather is a bit hot!"])
    source_lang: str = Field(..., examples=["en"], description="ISO 639-1 code")
    target_lan: str = Field(..., examples=["fr"], description="ISO 639-1 code")

class TranslationResponse(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    latency_ms: float

class LanguagesResponse(BaseModel):
    """
    Output schema for the get/languages path,
    shows a mapping of ISO-639-1 codes to NLLB Flores-200 language code used by model. 
    """
    supported_languages: dict[str,str]


global_batcher = BatchingTranslator()


@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Function is decorated with asynccontextmanager so FastAPI knows how to 
    handle the running of the server. Fast api will perform the warmup and batching start
    when the server boots. (downloading weights if first ever start up)

    Args:
        app (FastAPI): _description_
    """
    # before accepting requests load the model. 
    translation.warm_up()

    # event loop has the worker loop attached to it here, runs constantly in the background
    await global_batcher.start()
    # function then releases control of the server completely to fastapi until shutdown
    yield 
    # only when the server is shutdown does this command run, new requests are refused and catches shutdown
    # errors so it fails sensibly
    await global_batcher.stop()


app = FastAPI(title="Translation Service", 
              description="Mutlilingual translation API using the facebook/nllb-200-distilled-600m model", 
              version="1.0.0", 
              lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/languages", response_model=LanguagesResponse)
def get_languages() -> LanguagesResponse:
    return LanguagesResponse(supported_languages=NLLB_LANG_MAP)

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(trans_request: TranslationRequest, request: Request)-> TranslationResponse:

    # should implement guardrails here for the input, check it doesn't break guidelines, have unsupported languages