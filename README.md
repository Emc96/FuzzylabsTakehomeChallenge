# Translation Service

A RESTful translation API built with FastAPI and a Hugging Face
`transformers` model, submitted for the Fuzzy Labs take-home challenge
(Task A).

## Problem statement

Serve a small pre-trained language model as a translation endpoint:
take a string and a language pair, return the translated string. Beyond
the core endpoint, the brief asks for a solution that considers how it
would behave in practice — multiple languages, concurrent calls,
monitoring, evaluation, feedback, and safeguarding.

## Architecture

```
Client
  │  POST /translate {text, source_lang, target_lang}
  ▼
FastAPI (application/main.py)
  │  1. validate input (application/guardrails.py)
  │  2. queue job, await result (app/batching.py)
  ▼
BatchingTranslator 
  │  groups same-language-pair jobs into one batch,
  │  runs inference in a thread pool so the event loop never blocks
  ▼
translation.translate_batch (application/translation.py)
  │  tokenise → model.generate() → decode
  ▼
facebook/nllb-200-distilled-600M
```

## Model choice

**`facebook/nllb-200-distilled-600M`** — a single multilingual model covering 200
languages via `src_lang`/`tgt_lang` codes, rather than a family of
per-pair models (e.g. `Helsinki-NLP/opus-mt-en-fr`, `opus-mt-en-de`).

I chose one multilingual model over per-pair models because:
- Supporting a new language pair is a config change (add a code to
  `NLLB_LANG_MAP`), not a new model to download, load, and manage
  in memory. 
- Managing many individually tuned language models can become complex to manage test and deploy especially if they update at different frequencies. Would need routing logic for each model. Not practical in this time frame. 

The trade-off: `"facebook/nllb-200-distilled-600M"` is lower quality than a dedicated
model for any single pair, and non-English-centric pairs (e.g. fr→de) go
through a shared multilingual representation rather than a model tuned
for that specific pair. For a production system I'd benchmark both against the target use case's actual language pairs before committing. 

## API

| Endpoint     | Method | Description                                                                                     |
| ------------ | ------ | ----------------------------------------------------------------------------------------------- |
| `/translate` | POST   | `{text, source_lang, target_lang}` to `{translated_text, source_lang, target_lang, latency_ms}` |
| `/languages` | GET    | List of supported language codes                                                                |
| `/health`    | GET    | Liveness check                                                                                  |

Interactive docs (via FastAPI's auto-generated OpenAPI schema) are
available at `/docs` once the service is running.

## Further questions addressed

**1. Multiple language options**: handled by design: one multilingual
model, `NLLB_LANG_MAP` in `application/config.py` controls what's exposed.
Adding a language is one line, no retraining or new model, or adding a new model to the translation logic. 

**2. Continuous evaluation**:  `evaluate/evaluate.py` & `evalualte/testset.jsonl`.
A small held-out test set scored with BLEU (`sacrebleu`). Designed to run in CI on every model/config change (fail the build below a threshold) and periodically against sampled production traffic to catch drift. The bundled test set is deliberately tiny (6 sentences) as a demonstration.  Production would use something like the FLORES-200 (https://huggingface.co/datasets/Muennighoff/flores200) for broader coverage per language pair. 

**3. Concurrent calls**: `application/batching.py`. Two things are needed to
handle concurrent requests well: (a) inference must not block the event
loop, and (b) ideally concurrent requests are batched for GPU/CPU
efficiency rather than processed one at a time. `BatchingTranslator` does
both requests are queued, grouped by language pair within a short
window (default 50ms or 8 requests, whichever comes first), and run
through the model in a single `generate()` call. Managing latency and resource utilisaiton.  

**6. Guarding against inappropriate input/output**: 
`application/guardrails.py`, applied to both the incoming text and the model's output. Small example of what can be done (length limit + keyword patterns). see Limitations for the production version of this.
## Running it
### Using uv
If you have uv installed then you can use uv to easily recreate the environment, running: 
```bash
uv sync 
```
Then run it with: 
```bash
uv run fastapi run main.py
```
Or with automatic reloading 
```bash
uv run fastapi dev main.py
```

### Using Pip
```bash
pip install -r requirements.txt

# Mac/Linux 
source .venv/bin/activate

# Windows 
.venv\Scripts\activate
```
Then run the server with: 
```bash
fastapi run main.py
```
Or with automatic reloading:
```bash
fastapi dev main.py
```

First startup downloads the model from Hugging Face (about 1GB with
dependencies). this needs network access to huggingface.co and takes a
minute or two. Then visit `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning Fuzzy Labs!", "source_lang": "en", "target_lang": "fr"}'
```

###  Using Docker

```bash
docker build -t translation-service .
docker run -p 8000:8000 translation-service
```
Could also mount a volume so that the model doesn't have to redownload every time the server is started: 
```bash
docker run -p 8000:8000 -v hf_cache:/root/.cache/huggingface translation-api
```
### Tests

```bash
pytest test_guardrails.py
```

These tests check that the guardrails logic is applied correctly. 

### Evaluation

```bash
python -m eval.evaluate
```

Requires the real model. Prints BLEU per language pair
in the test set. 

## Limitations / what I'd do with more time

- **Guardrails are a keyword-match.** Production would use a moderation model (e.g. a small classifier) or hosted moderation API, applied to both input and output, with the rejection reason logged for review rather than only returned to the user.
- **Batching only groups same-language-pair requests.** Under mixed-pair load this reduces the effective batch size. An alternative is padding/masking to batch across pairs, I would construct a matrix of `decoder_input_ids` where the starting token of each sequence corresponds to its specific target language. This allows the model to process multiple target languages in a single forward pass. I would also look to build a sequence size matcher, don't allow large sequences batched with small ones which hurts the response time of the small sequences. Batch similar size sequences together to improve response time.
- **API Tests.** There are no API tests in this repo, just because of running out of time. I would have integration tests implemented with pytest and FastAPIs TestClient so I can evaluate the concurrent request handling, queue management, test responses, ensure the API rejects unsupported languages and inputs etc. 
- **Eval test set is tiny and hand-picked**, not sourced from a standard benchmark. Fine as a demonstration of the mechanism, not as a real quality bar. I would use some established large benchmark datasets to actually evaluate performance against, especially those low resourced languages because multi-lingual models struggle with these languages the most therefore, monitoring the performance of them is key to ensure good responses. Utilise standardised benchmark datasets such as FLORES-200 to evaluate the model against a range of languages and topics that will present a good picture of model performance. BLEU is purely lexical and so it only measures word overlap which is okay as a reporting metric but it can't be the only one as models can output responses that are correct but not lexically similar ("How are you?", "Whats up?"). Utilise something like COMET which is a more complex metric utilising neural networks to evaluate responses on fluency, context and actual meaning. Expandind metrics and utilisng more complex types will again build a better monitoring suite of the model in production.  
- **Monitoring in Production**. Currently there is no real monitoring of the model happening. Would integrate either MLflow or Evidently AI generative AI monitoring to the API to at least be able to surface issues and performance of the responses from the LLM. Build on the latency being recorded and record batch sizes, guardrail rejection rates, number of requests, number of requests per language. Have it behind a /metrics endpoint. The metrics used for evaluation would also be greatly improved upon. These will be good steps in helping to detect data drift and response degradation. 
- **Incorporating end-user feedback** — I'd add an optional`translation_id` to each response and a `POST /feedback {translation_id, rating, corrected_text}` endpoint, logging feedback alongside the original input/output. That data has two uses: flagging low-rated language pairs for the evaluation set and then fine-tuning data. I'd not retrain on raw feedback without review, since that's an easy way to let a small number of bad faith submissions poison the model.