"""Memory-conscious CPU inference for resource-constrained Streamlit deployment.

Only one Hugging Face pipeline is resident at a time. A request lock serializes
model transitions; model files remain cached on disk between stages.
"""
from __future__ import annotations

import gc
import logging
import threading

logger = logging.getLogger("finnews.runtime")


class ModelRuntime:
    def __init__(self):
        self.request_lock = threading.RLock()
        self._pipeline = None
        self._key = None

    def release(self):
        """Drop the last model before constructing the next one."""
        self._pipeline = None
        self._key = None
        gc.collect()
        # On Linux, return unused libc heap pages when available.
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def load(self, task: str, model_id: str):
        with self.request_lock:
            key = (task, model_id)
            if self._pipeline is not None and self._key == key:
                return self._pipeline
            self.release()
            from transformers import pipeline
            options = {"device": -1, "model_kwargs": {"low_cpu_mem_usage": True}}
            if task == "text-classification":
                options["top_k"] = None
            elif task == "token-classification":
                options["aggregation_strategy"] = "first"
            self._pipeline = pipeline(task, model=model_id, tokenizer=model_id, **options)
            self._key = key
            logger.info("Loaded pipeline task=%s model=%s", task, model_id)
            return self._pipeline


def run_intelligence(article: str, runtime: ModelRuntime, models: dict, progress=None):
    """Use the same inference logic as the tested app, with serialized model stages."""
    import time
    from intelligence import (
        classify_text, company_sentiments, extract_companies,
        extractive_summary, generated_summary,
    )
    from intelligence import split_sentences

    def notify(message):
        if progress is not None:
            progress(message)

    diagnostics = {"entity_model": models["entity"], "summary_model": models["summary"]}
    with runtime.request_lock:
        try:
            notify("Identifying companies in your news...")
            started = time.perf_counter()
            ner = runtime.load("token-classification", models["entity"])
            companies, sentences, entity_coverage = extract_companies(article, ner)
            diagnostics["entity_seconds"] = time.perf_counter() - started
            del ner
            runtime.release()

            notify("Understanding the financial sentiment...")
            started = time.perf_counter()
            classifier = runtime.load("text-classification", models["sentiment"])
            diagnostics["sentiment_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            if companies:
                rows = company_sentiments(article, companies, sentences, classifier)
                overall = None
            else:
                rows = []
                overall = classify_text(article, classifier)
            diagnostics["sentiment_seconds"] = time.perf_counter() - started
            del classifier
            runtime.release()

            notify("Preparing your news summary. Initial model downloads may take a few minutes...")
            started = time.perf_counter()
            summary_coverage = {"truncated": False}
            method = "generated"
            try:
                summarizer = runtime.load("summarization", models["summary"])
                summary, summary_coverage = generated_summary(article, summarizer)
                del summarizer
            except Exception:
                logger.exception("Generated summary unavailable; selecting source sentences")
                # Fallback is disclosed in the interface, never counted as model success.
                summary = extractive_summary(article, sentences or split_sentences(article))
                method = "extractive"
            diagnostics["summary_seconds"] = time.perf_counter() - started
            return {"companies": rows, "overall": overall, "summary": summary,
                    "summary_method": method, "summary_coverage": summary_coverage,
                    "entity_coverage": entity_coverage, "diagnostics": diagnostics}
        finally:
            runtime.release()
