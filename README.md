# FinNews Insight

**Financial news, distilled into perspective — company intelligence, sentiment signals, and the context that matters.**

FinNews Insight turns an uploaded English financial-news document into detected company names, financial sentiment and a concise summary. This is an ISOM5240 coursework application, not a trading system or investment recommendation.

## User workflow

1. Upload a TXT, Markdown, DOCX or text-based PDF article.
2. Select **Click to get insights**.
3. Read **Company**, **Financial sentiment** and **News summary**.
4. Expand the supporting sentences or original text when needed.

The customer interface does not show model scores, model paths or inference timers. Scanned PDFs require OCR before upload. The file limit is 10 MB and the PDF limit is 30 pages. Model coverage is limited for very long articles and the interface warns when processing is partial.

## Hugging Face model pipelines

| Task | Pipeline | Model |
| --- | --- | --- |
| Company/organization recognition | `token-classification` | `dslim/distilbert-NER` |
| Financial sentiment | `text-classification` | The saved student-fine-tuned FinBERT, configured via `SENTIMENT_MODEL` |
| News summary | `summarization` | `sshleifer/distilbart-cnn-12-6` |

The sentiment model is based on `ProsusAI/finbert`. **This application never silently substitutes the public base model for the student's fine-tuned model.** Configure a valid saved-model directory or the actual published Hugging Face repository before testing inference.

The sentiment classifier has three trained labels: negative, neutral and positive. `Mixed` is a heuristic presentation flag for conflicting strong predictions across article chunks; `Unclear` indicates insufficient company-specific attribution, not a trained fourth class.

NER recognizes organizations, which can include non-company entities and incorrect or missed names. For one detected company, sentiment summarizes the article's overall tone. For multiple companies, the application selects financial sentences mentioning each organization without another detected organization. Shared or unassignable statements are not treated as confidently company-specific predictions. This is not a dedicated aspect-based sentiment or causal-impact model.

The generated summary can contain errors; readers should verify material figures against the source article. If summary generation fails, an explicitly labeled extractive fallback selects source sentences. A fallback is not counted as a successful generated summary.

## Local execution

Use Python **3.11**. The Linux deployment requirements use CPU-only PyTorch; no GPU is required for inference.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export SENTIMENT_MODEL=/absolute/path/to/finbert-finetuned
streamlit run app.py
```

For a published model, replace the path with the actual Hugging Face Model ID. On Windows, a separate platform-appropriate PyTorch installation may be needed; the supplied pinned requirements target Linux/Streamlit Cloud.

## Streamlit Community Cloud

1. Upload the contents of this directory to the root of a GitHub repository. Preserve the `.streamlit/` folder.
2. At `share.streamlit.io`, select **Create app** and choose the repository and `main` branch.
3. Set the main file path to **`app.py`** and select **Python 3.11** in Advanced settings.
4. Copy the settings from `.streamlit/secrets.example.toml` into the app's Secrets settings. Replace the model placeholder with the **actual** student-fine-tuned repository.
5. Deploy, run at least one full article through all model stages, and inspect Cloud logs if startup or inference fails.
6. Complete the deployment-specific accuracy experiment and capture actual Cloud screenshots before claiming submission readiness.

A Colab or `trycloudflare.com` preview is not the required Streamlit Cloud deployment. If the saved model is only in Google Drive, Cloud cannot read a `/content/drive/...` path. Publish the same weights to Hugging Face, or use an explicitly configured, independently verified alternative model-delivery route. A model route does not replace the coursework requirement to provide an HF Model URL.

For a public model, anonymous downloading is normally supported; creating and publishing a repository still requires authorized HF access. Store any private-model token only in Cloud Secrets, never in source files.

## Deployment resource strategy

`model_runtime.py` holds **one pipeline in memory at a time** and serializes analysis requests. NER is unloaded before sentiment, and sentiment is unloaded before summarization. Model files remain cached on disk, but are reloaded into memory for each request. This reduces simultaneous model residency at the cost of additional loading time. It does not guarantee that every Community Cloud resource tier will support the summary model; actual Cloud testing remains mandatory.

All three chosen models are unchanged from the tested pipeline. Do not change the summary or sentiment model silently just to make deployment fit. Any model substitution must be recorded and re-evaluated.

## Files

- `app.py`: English Streamlit entrypoint and presentation.
- `intelligence.py`: entities, evidence scope, sentiment aggregation and summary logic.
- `news_utils.py`: extraction, token-aware chunking and label checks.
- `model_runtime.py`: serialized, memory-conscious model lifecycle.
- `styles.css`: fixed light theme and contrast styling.
- `.streamlit/config.toml`: application theme and file-size settings.
- `.streamlit/secrets.example.toml`: example configuration, not actual credentials.
- `tests_v2.py`: software regression fixtures; not model-accuracy measurements.
- `validate_model.py`: validate and hash saved model files for deployment consistency.

Training and experiment notebooks, datasets, final report, Excel results, slides, video and model weights belong in the separate coursework submission folders. They should not be confused with this deployable application directory.

## Verification

```bash
python -m py_compile app.py intelligence.py news_utils.py model_runtime.py validate_model.py
python tests_v2.py
python validate_model.py /path/to/finbert-finetuned --output model_manifest.json
```

Regression tests use synthetic fixtures and must never be reported as empirical application accuracy. Integration testing with public base FinBERT verifies software plumbing only, not the student's saved weights.

## Model sources and deployment references

- https://huggingface.co/ProsusAI/finbert
- https://huggingface.co/dslim/distilbert-NER
- https://huggingface.co/sshleifer/distilbart-cnn-12-6
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app

Research assistance only. Financial sentiment is not a prediction of future share-price movements or personalized investment advice.
