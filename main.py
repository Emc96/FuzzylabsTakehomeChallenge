import logging 
import time 

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request 
from pydantic import BaseModel, Field 

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
    shows a mappign of ISO-639-1 code to NLLB Flores-200 language code used by model. 
    """
    supported_languages: dict[str,str]