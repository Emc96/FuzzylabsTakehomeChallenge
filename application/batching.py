"""
This script contains the logic around batching multiple requests into groups
of the same source and target language and then passed to the model as one,
utilise parrallism of transformer models and lower the impact of the model
generation on blocking the API from getting more requests/managing worker queue.

"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from . import translation
from .config import MAX_BATCH_SZIE, MAX_WAIT_MS


logger = logging.getLogger(__name__)


@dataclass
class TranslationJob:
    """
    Use dataclass here instead of pydantic, it has to be a lightweight storage of
    requests/jobs because this object is used in the event loop don't need pydantics relatively heavy
    validation checks type coercion etc in the event loop.
    There could theoreticaly 1000s of translationjobs objects created as the API gets pinged.
    Pydantic would force this object into json object which would error in the event loop, event loop is
    not expecting a json object
    Validation is handled at the API endpoint level not the tranlsation layer.
    """

    text: str
    source_lang: str
    target_lang: str
    future: asyncio.Future[str] = field(compare=False)


class BatchingTranslator:
    def __init__(
        self, max_batch_size: int = MAX_BATCH_SZIE, max_wait_ms: float = MAX_WAIT_MS
    ):

        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        # create the request queue here, hint at what objects will fill the queue too
        self.queue: asyncio.Queue[TranslationJob] = asyncio.Queue
        # going to wrap the coroutine worker into the python event loop. This allows python/server
        # to check if the current coroutinte is running, pending, done or cancelled.
        # this will be created in a method
        self.worker_task: asyncio.Task | None = None

    async def worker_loop(self):

        while True:
            batch = self.collect_batch()
            await self.process_batch(batch)

    async def start(self):
        """ """
        # need to create worker loop for get batches and executing them
        self.worker_task = asyncio.create_task()

    async def stop(self):

        return

    async def submit(self, text: str, source_lang: str, target_lang: str):
        """
        New requests from the API are submitted to the translation layer through this method.
        It uses the TranslationJob dataclass to create translation objects and then adds them
        to the back of the queue for the loop to pickup and batch accordingly to their source
        & target language.

        Args:
            text (str): Text to translate
            source_lang (str): source language
            target_lang (str): target language
        """
        # get the asyncio event loop, futures have to attached to specific loops to be monitored
        # for changes to their status
        loop = asyncio.get_running_loop()

        # need to create that future object the translationjob object requires so it can be handled correctly.
        job_future: asyncio.Future[str] = loop.create_future()

        # with the future object created, bundle the info from the request together and add it to the back of the queue
        # it will be picked up by the collect batch method.
        await self.queue.put(TranslationJob(text, source_lang, target_lang, job_future))
        # once the process batch method has completed the specific request the job future is released
        # the event loop can pick it up and delvier the results back to the endpoint
        return await job_future

    async def collect_batch(self) -> list[TranslationJob]:
        """
        Method will collect batches together, and based on whether the max batch size is met
        or the max wait time will then move the batch onto processing.
        Balancing user wait time and utilisation of resources here.

        Returns:
            list[TranslationJob]: _description_
        """
        # wait for the queue to be filled with a first request, no point working without one
        # if nothing there hand control back to event loop to get one
        first = await self.queue.get()

        batch = [first]

        # create a deadline for the max_wait_time
        # fastapit understands seconds, convert from ms
        deadline = time.monotonic() + self.max_wait_ms / 1000

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(job)
            # stop the api from crashing if no requests are made in the batch timeframe
            except asyncio.TimeoutError:
                break

        return batch

    async def process_batch(self, batch: list[TranslationJob]):
        """
        Once the batch is gathered, execute the processing
        Gather pair of language requests together based on source & target language
        Model can only process one language at a time so need to group them by unique language parings

        Args:
            batch (list[TranslationJob]): Batch from collect_batch method, list of translationjob objects.
        """
        # define what the language pair groups are, the key will be a tuple of
        # source language, target language and the value will be a list of the translsation jobs
        # that match that pair of languages
        groups: dict[tuple[str, str], list[TranslationJob]] = {}

        for job in batch:
            language_pair = (job.source_lang, job.target_lang)
            # if language pair doesn't exist, needs adding to the groups
            if language_pair not in groups:
                groups[language_pair] = []
            # each language pair task is added to their language group
            groups[language_pair].append(job)

        loop = asyncio.get_running_loop()

        for (src, tgt), jobs in groups.items():

            texts = [translate_job.text for translate_job in jobs]

            try:
                # this block is where the blockage to the app can happen.
                # await releases the event thread to go back and handle new requests until this
                # is complete.
                results = await loop.run_in_executor(
                    None, 
                    translation.translate_batch, 
                    texts, 
                    src, 
                    tgt
                )

                for job, result in zip(jobs, results):
                    if not job.future.done():
                        # the future internal state is set to finished,
                        # submit method can now continue, translation is placed in the futures of the 
                        # translation job object
                        job.future.set_result(result)


            except Exception as e:
                logger.exception("Batch translation failed for %s->%s", src, tgt)
                for job in jobs:
                    # if there is an error on the translation job update the future so 
                    # app can still run 
                    if not job.future.done():
                        job.future.set_exception(e)