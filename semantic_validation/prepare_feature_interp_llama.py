import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

from transformers import AutoTokenizer


# -----------------------------
# Config
# -----------------------------
OPENAI_MODEL = "gpt-5.1"
INPUT_JSON = "result/fineweb_feature_activations_llama.json"   # your full file
OUTPUT_JSONL = f"feature_interpret_requests_{OPENAI_MODEL}.jsonl" # batch input
OUTPUT_PREVIEW_CSV = f"feature_snippet_preview_{OPENAI_MODEL}.csv"

MODEL_FOR_TOKENIZER = None  # if None, uses data["model"] from input json
WINDOW_TOKENS = 64          # ±64 tokens recommended
EVIDENCE_SNIPPETS_PER_FEATURE = 12
DISPLAY_SNIPPETS_PER_FEATURE = 3



# -----------------------------
# Utility: snippet extraction
# -----------------------------
_whitespace_re = re.compile(r"\s+")

def normalize_for_dedup(s: str) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = _whitespace_re.sub(" ", s).strip()
    return s

def extract_token_window(
    text: str,
    tok_idx: int,
    tokenizer,
    window: int = 64,
    add_special_tokens: bool = True
) -> str:
    """
    Extract a window around token index tok_idx: [tok_idx-window, tok_idx+window].
    Uses the same tokenizer family as the underlying LM for clean detokenization.
    """
    enc = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_tensors=None
    )
    ids = enc["input_ids"]
    n = len(ids)

    if n == 0:
        return text[:400]

    # Clamp token index if it is out of bounds (can happen if text changed or preprocessing differs)
    if tok_idx < 0:
        tok_idx = 0
    if tok_idx >= n:
        tok_idx = n - 1

    lo = max(0, tok_idx - window)
    hi = min(n, tok_idx + window + 1)

    snippet = tokenizer.decode(ids[lo:hi], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    snippet = snippet.strip()

    # Add ellipses if we truncated context
    if lo > 0:
        snippet = "… " + snippet
    if hi < n:
        snippet = snippet + " …"

    return snippet


# -----------------------------
# Schema + prompt
# -----------------------------
MFT_DEFS = """Moral Foundations Theory (MFT) definitions:
- Care/harm: dislike others’ suffering; kindness, gentleness, nurturance vs cruelty, violence.
- Fairness/cheating: justice, rights, autonomy vs fraud, exploitation, cheating.
- Loyalty/betrayal: group allegiance, patriotism, self-sacrifice vs betrayal, treason, disloyalty.
- Authority/subversion: respect for legitimate authority, leadership/followership, traditions vs defiance, disrespect, subversion.
- Sanctity/degradation: purity, elevation above the carnal, disgust sensitivity vs degradation, contamination, depravity.
"""

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "short_label": {"type": "string"},
        "long_description": {"type": "string"},
        "mft_alignment": {
            "type": "string",
            "enum": ["care", "fairness", "loyalty", "authority", "sanctity", "none"]
        },
        "mft_polarity": {
            "type": "string",
            "enum": ["virtue", "vice", "mixed", "none"]
        },
        "rationale": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 6
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["short_label", "long_description", "mft_alignment", "mft_polarity", "rationale", "evidence_ids", "confidence"]
}

def build_prompt(feature_meta: Dict[str, Any], snippets: List[str]) -> str:
    """
    Conservative morality-aware interpretation:
    - Always describe what the feature detects first (topic/style/intent)
    - Only assign an MFT alignment if strongly supported by evidence
    """
    lines = []
    lines.append("You are interpreting a sparse autoencoder (SAE) feature from an LLM.")
    lines.append("Your job: infer the most likely semantic pattern that triggers the feature, based ONLY on the evidence snippets.")
    lines.append("")
    lines.append("Instructions:")
    lines.append("1) First, describe the dominant pattern neutrally (topic, style, rhetorical function, or social behavior).")
    lines.append("2) Then, OPTIONAL: map it to Moral Foundations Theory category if supported. Otherwise output mft_alignment='none'.")
    lines.append("3) Do not force morality. Many features are not moral.")
    lines.append("4) Provide a short label (5–10 words) and a 1–2 sentence long description.")
    lines.append("5) Cite evidence_ids (indices of snippets) that justify your decision.")
    lines.append("")
    lines.append(MFT_DEFS)
    lines.append("Feature metadata (do not overfit to this):")
    lines.append(json.dumps(feature_meta, ensure_ascii=False))
    lines.append("")
    lines.append("Evidence snippets (index: text):")
    for i, s in enumerate(snippets):
        lines.append(f"[{i}] {s}")
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    seen_pairs = set()

    inp = Path(INPUT_JSON)
    data = json.loads(inp.read_text())

    model_name = MODEL_FOR_TOKENIZER or data.get("model")
    if not model_name:
        raise ValueError("No tokenizer model specified and input JSON missing top-level 'model' field.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    features = data["features"]
    preview_rows: List[Dict[str, Any]] = []

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for f in features:
            # ---- Deduplicate requests by (layer, feature_id) ----
            pair = (int(f["layer"]), int(f["feature_id"]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            feature_meta = {
                "layer": int(f["layer"]),
                "feature_id": int(f["feature_id"]),
                "sae_layer": int(f.get("sae_layer", f["layer"])),
                "foundation_hint": f.get("foundation", "unknown"),
                "cosine_similarity": float(f.get("cosine_similarity", 0.0)),
                "rank_in_layer": int(f.get("rank_in_layer", -1)),
            }

            # 1) window snippets
            raw_acts = f["top_activations"]
            snippets_all: List[str] = []
            for a in raw_acts:
                text = a["text"]
                tok_idx = int(a["max_activation_token_index"])
                snippets_all.append(extract_token_window(text, tok_idx, tokenizer, window=WINDOW_TOKENS))

            # 2) deduplicate snippets within feature
            deduped: List[str] = []
            seen = set()
            for s in snippets_all:
                key = normalize_for_dedup(s)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(s)

            evidence = deduped[:EVIDENCE_SNIPPETS_PER_FEATURE]
            display = evidence[:DISPLAY_SNIPPETS_PER_FEATURE]

            prompt = build_prompt(feature_meta, evidence)

            batch_line = {
                "custom_id": f"layer{feature_meta['layer']}_fid{feature_meta['feature_id']}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": OPENAI_MODEL,
                    "instructions": "Return only valid JSON matching the provided schema.",
                    "input": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "sae_feature_interpretation",
                            "schema": JSON_SCHEMA,
                            "strict": True
                        }
                    }
                }
            }
            fout.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

            preview_rows.append({
                **feature_meta,
                "snippet_1": display[0] if len(display) > 0 else "",
                "snippet_2": display[1] if len(display) > 1 else "",
                "snippet_3": display[2] if len(display) > 2 else "",
            })

    pd.DataFrame(preview_rows).to_csv(OUTPUT_PREVIEW_CSV, index=False)
    print(f"Wrote batch file: {OUTPUT_JSONL}")
    print(f"Wrote preview CSV: {OUTPUT_PREVIEW_CSV}")



if __name__ == "__main__":
    main()
