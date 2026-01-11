import json
import pandas as pd

gpt_model = '5.1'
BATCH_OUTPUT_JSONL = f"batch_output_{gpt_model}.jsonl"
OUT_CSV = f"feature_interpretations_{gpt_model}.csv"

def extract_structured_json(body: dict):
    """
    Extract the JSON returned by Structured Outputs from a Responses API payload.
    """
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    return json.loads(c.get("text", ""))
    raise ValueError("Could not find structured JSON in response body.")

rows = []
with open(BATCH_OUTPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        body = (obj.get("response") or {}).get("body") or {}

        try:
            parsed = extract_structured_json(body)
            rows.append({"custom_id": custom_id, **parsed})
        except Exception as e:
            rows.append({
                "custom_id": custom_id,
                "parse_failed": True,
                "error": str(e),
                "raw_preview": json.dumps(body)[:1500],
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)
print("Parse failures:", int(df.get("parse_failed", False).sum()) if "parse_failed" in df.columns else 0)
