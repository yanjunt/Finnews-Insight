"""Business logic: organization extraction, evidence-scoped sentiment and summaries.

NER recognizes organizations, not verified legal company identities. Sentiment is
sentence-based, not an aspect-based relation model. Ambiguous attribution is surfaced.
"""
from __future__ import annotations

import re
from collections import Counter

from news_utils import LABEL_NAMES, clean_text, probability_rows, token_chunks

NON_COMPANY_TERMS = {
    "eps", "gaap", "non-gaap", "cfo", "ceo", "usd", "eur", "us", "u.s.",
    "sec", "nyse", "nasdaq", "s&p", "q1", "q2", "q3", "q4", "california",
    "company", "the company", "group", "the group", "board of directors",
    "services", "products", "operating cash flow", "revenue",
}
FINANCIAL_WORDS = re.compile(
    r"\b(revenue|sales|profit|loss|earnings|cash flow|dividend|guidance|"
    r"forecast|growth|decline|debt|acquisition|layoffs|bankruptcy|shares)\b", re.I
)


def article_body(text):
    """Drop source/footer metadata, not the news itself."""
    text = clean_text(text)
    text = re.split(r"(?im)^\s*(?:Source|Sources|Selected original excerpt)\s*[:;]", text)[0]
    lines = [line for line in text.splitlines()
             if not re.match(r"^\s*https?://", line)]
    text = "\n".join(lines)
    # Newswire datelines in all caps can otherwise merge a location into an ORG.
    text = re.sub(r"(?m)^[A-Z][A-Z .'-]+,\s*[A-Z][A-Z .'-]+(?=\s+[A-Z][a-z])\s*", "", text)
    return clean_text(text)


def normalized_name(name):
    name = re.sub(r"(?:['’]s)$", "", name.strip())
    name = re.sub(r"\s+", " ", name).strip(" ,;:.\"'“”")
    return name


def company_key(name):
    name = normalized_name(name).lower()
    # Merge simple suffix variants. Do not invent ticker/company name mappings.
    name = re.sub(r",?\s+(?:incorporated|inc\.?|corporation|corp\.?|limited|ltd\.?)$", "", name)
    return name.strip()


def split_sentences(text):
    return [part.strip() for part in re.split(r'(?<=[.!?])\s+(?=[A-Z"“])|\n\n+', text) if part.strip()]


def extract_companies(text, ner_pipeline, maximum=8):
    ner_text = text[:40000]
    chunks, ner_coverage = token_chunks(ner_text, ner_pipeline.tokenizer, max_length=384, max_chunks=128)
    counts = Counter()
    display_names = {}
    aliases = {}
    offsets = {}
    for chunk in chunks:
        for ent in ner_pipeline(chunk):
            if ent.get("entity_group") != "ORG":
                continue
            start, end = ent.get("start", 0), ent.get("end", 0)
            if start > 0 and chunk[start - 1].isalnum():
                continue
            if end < len(chunk) and chunk[end:end + 1].isalnum():
                continue
            name = normalized_name(chunk[start:end] if end > start else ent["word"])
            key = company_key(name)
            if not key or key in NON_COMPANY_TERMS or len(name) < 2:
                continue
            if re.fullmatch(r"[\d\W]+", name):
                continue
            counts[key] += 1
            display_names.setdefault(key, name)
            aliases.setdefault(key, set()).add(name)
            offsets.setdefault(key, len(offsets))
    ordered = sorted(counts, key=lambda k: (-counts[k], offsets[k]))
    companies = []
    for key in ordered[:maximum]:
        companies.append({"name": display_names[key], "key": key,
                          "aliases": sorted(aliases[key]), "mentions": counts[key]})
    sentences = split_sentences(ner_text)
    return companies, sentences, {"truncated": len(text) > 40000 or ner_coverage["truncated"],
                                  "company_limit": len(ordered) > maximum}


def mentions_company(sentence, company):
    for alias in company["aliases"]:
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence, re.I):
            return True
    # Also recognize the suffix-free form if it actually appears in the text.
    return bool(re.search(r"(?<!\w)" + re.escape(company["key"]) + r"(?!\w)", sentence, re.I))


def evidence_for_companies(companies, sentences):
    output = {c["key"]: {"unique": [], "shared": []} for c in companies}
    for sentence in sentences:
        present = [c for c in companies if mentions_company(sentence, c)]
        if not present:
            continue
        slot = "unique" if len(present) == 1 else "shared"
        for company in present:
            output[company["key"]][slot].append(sentence)
    return output


def classify_text(text, pipe):
    chunks, coverage = token_chunks(text, pipe.tokenizer, max_length=128, max_chunks=32)
    if not chunks:
        raise ValueError("There is no article text to analyze.")
    weighted = {label: 0.0 for label in LABEL_NAMES}
    total = 0
    individual = []
    for chunk in chunks:
        weight = max(1, len(pipe.tokenizer.encode(chunk, add_special_tokens=False)))
        scores = {r["label"]: r["score"] for r in probability_rows(
            pipe(chunk, truncation=True, max_length=128)
        )}
        individual.append(scores)
        for label in LABEL_NAMES:
            weighted[label] += scores[label] * weight
        total += weight
    scores = {k: v / total for k, v in weighted.items()}
    label = max(scores, key=scores.get)
    # Confidence is internal only. Mixed is a presentation flag, not a trained fourth class.
    strong_positive = any(x["positive"] >= 0.70 for x in individual)
    strong_negative = any(x["negative"] >= 0.70 for x in individual)
    if strong_positive and strong_negative:
        label = "mixed"
    return {"sentiment": label, "scores": scores, "coverage": coverage}


def company_sentiments(article, companies, sentences, pipe):
    evidence = evidence_for_companies(companies, sentences)
    results = []
    for company in companies:
        row = dict(company)
        item = evidence[company["key"]]
        if len(companies) == 1:
            # One detected organization: approximate article-level association, not coreference.
            prediction = classify_text(article, pipe)
            row.update(prediction, scope="Article-level sentiment",
                       note="Overall news tone for the main detected company.",
                       evidence=item["unique"][:3])
        else:
            relevant = [s for s in item["unique"] if FINANCIAL_WORDS.search(s)]
            if not relevant:
                row.update(sentiment="unclear", scores=None,
                           scope="Company-specific sentiment unclear",
                           note="The article does not provide enough separately attributable financial text for this company.",
                           evidence=item["shared"][:2])
            else:
                prediction = classify_text(" ".join(relevant), pipe)
                row.update(prediction, scope="Sentiment of company-relevant sentences",
                           note="Based on sentences that name this company without another detected company.",
                           evidence=relevant[:3])
                if item["shared"]:
                    row["note"] += " Shared-company sentences were excluded to avoid ambiguous attribution."
        results.append(row)
    return results


def extractive_summary(article, sentences):
    """Fallback that selects real sentences rather than fabricating a generated summary."""
    candidates = [s for s in sentences if len(s.split()) >= 8]
    if not candidates:
        return article[:700]
    indexed = list(enumerate(candidates))
    selected = sorted(indexed, key=lambda x: (-len(FINANCIAL_WORDS.findall(x[1])), x[0]))[:3]
    return " ".join(s for _, s in sorted(selected))


def generated_summary(article, summarizer):
    chunks, coverage = token_chunks(article, summarizer.tokenizer, max_length=850, max_chunks=2)
    parts = []
    for chunk in chunks:
        n = len(summarizer.tokenizer.encode(chunk, add_special_tokens=False))
        if n < 45:
            parts.append(chunk)
            continue
        maximum = min(110, max(25, n // 2))
        minimum = min(30, maximum - 1)
        result = summarizer(chunk, max_length=maximum, min_length=minimum,
                            truncation=True, do_sample=False)
        parts.append(result[0]["summary_text"].strip())
    if not any(parts):
        raise ValueError("The summary model returned an empty result.")
    return " ".join(parts), coverage
