"""
Some simple guardrails that will guard against some simple inappropriate input/output
This will just have one function that will perform some simple checks on the texts
Can't have an empty input: return an error message
If the text exceeds the max input lenght: return an error message
if the message contains inappropriate text: return an error message.
In a production system there probably should be some sort of classification model monitoring
inputs/outputs or use some guardrail filtering API.
"""

import logging
import re

from .config import MAX_INPUT_LENGTH

blocked_patterns = [
    re.compile(r"\bhow to (make|build|source) a bomb\b", re.IGNORECASE),
    re.compile(
        r"\bhow to (make|build|source|synthesise) (poision|nerve agent)\b",
        re.IGNORECASE,
    ),
]

logger = logging.getLogger(__name__)


def validate_text(text: str, max_length: int = MAX_INPUT_LENGTH):

    if text is None or not text.strip():
        error = "Text cannot be empty"
        logger.warning("Issue with input/output %s", error)
        return error
    elif len(text) > max_length:
        logger.warning("Max length reached on input/output")
        return f"Text cannot exceed limit of {max_length} characters"

    for pattern in blocked_patterns:
        if pattern.search(text):
            logger.warning("Input/output contained blocked pattern: %s", pattern)
            return "Text contains disallowed content"
    return None
