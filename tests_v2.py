"""Software regression tests using fixtures, not measured financial model accuracy."""
from pathlib import Path
import hashlib
import os

from intelligence import article_body, company_sentiments, split_sentences
from model_runtime import ModelRuntime
from streamlit.testing.v1 import AppTest


class FixtureTokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, tokens, **kwargs):
        return " ".join(tokens)
    def num_special_tokens_to_add(self, pair=False):
        return 2


class FixtureClassifier:
    tokenizer = FixtureTokenizer()
    def __call__(self, text, **kwargs):
        if "fell" in text:
            values = [0.95, 0.03, 0.02]
        elif "increased" in text:
            values = [0.02, 0.03, 0.95]
        else:
            values = [0.05, 0.90, 0.05]
        return [{"label": k, "score": v} for k, v in zip(["negative", "neutral", "positive"], values)]


companies = [{"name": "Apple", "key": "apple", "aliases": ["Apple"], "mentions": 1},
             {"name": "Microsoft", "key": "microsoft", "aliases": ["Microsoft"], "mentions": 1}]
article = "Apple revenue increased. Microsoft profit fell."
rows = company_sentiments(article, companies, split_sentences(article), FixtureClassifier())
assert [(r['name'], r['sentiment']) for r in rows] == [('Apple', 'positive'), ('Microsoft', 'negative')]
shared = "Apple and Microsoft reported revenue growth."
rows_shared = company_sentiments(shared, companies, [shared], FixtureClassifier())
assert all(r['sentiment'] == 'unclear' for r in rows_shared)
assert 'CUPERTINO' not in article_body('CUPERTINO, CALIFORNIA Apple reported higher revenue.\n\nSource: Apple Newsroom.')
assert 'Newsroom' not in article_body('Apple reported higher revenue.\n\nSource: Apple Newsroom.')

# Render populated customer results without user model weights or network calls.
at = AppTest.from_file(str(Path(__file__).resolve().parent / 'app.py')).run(timeout=20)
assert not at.exception
model = os.getenv('SENTIMENT_MODEL', 'models/finbert-finetuned')
summary_model = os.getenv('SUMMARY_MODEL', 'sshleifer/distilbart-cnn-12-6')
entity_model = os.getenv('ENTITY_MODEL', 'dslim/distilbert-NER')
sig = hashlib.sha256(('' + model + summary_model + entity_model + '2.2').encode()).hexdigest()
at.session_state['signature'] = sig
at.session_state['analysis'] = {
    'companies': [rows[0]], 'overall': None,
    'summary': 'Apple revenue was $95.4 billion and EPS was $1.65. <script>bad</script>', 'summary_method': 'generated',
    'summary_coverage': {'truncated': False},
    'entity_coverage': {'truncated': False, 'company_limit': False},
    'diagnostics': {'sentiment_seconds': 7.64}, 'source': 'fixture.txt'
}
at.run(timeout=20)
assert not at.exception
visible = '\n'.join([x.value for x in at.markdown] + [x.value for x in at.caption])
assert 'Apple' in visible and 'Positive' in visible
assert '7.64' not in visible
assert len(at.metric) == 0 and len(at.text_input) == 0 and len(at.checkbox) == 0
assert any(x.value == 'News summary' for x in at.subheader)
assert at.button[0].label == 'Click to get insights'
assert '$95.4 billion' in visible and '$1.65' in visible
assert '&lt;script&gt;bad&lt;/script&gt;' in visible
assert '<script>bad</script>' not in visible
assert 'Financial news, distilled into perspective' in visible
runtime = ModelRuntime()
runtime._pipeline = object()
runtime._key = ('test', 'test')
runtime.release()
assert runtime._pipeline is None and runtime._key is None
print('PASS: separate company sentiment, shared-sentence uncertainty, metadata cleanup, English result cards and hidden technical metrics')
