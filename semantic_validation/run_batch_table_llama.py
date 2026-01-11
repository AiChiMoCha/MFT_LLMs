import json
from pathlib import Path
import pandas as pd
from openai import OpenAI
model = 'gpt-5.1'
BATCH_INPUT_JSONL = f"feature_interpret_requests_{model}.jsonl"
BATCH_OUTPUT_JSONL = f"batch_output_{model}.jsonl"  # downloaded output file content will go here
FINAL_TABLE_CSV = f"sae_feature_table_with_interpretations_{model}.csv"

client = OpenAI()

def submit_batch(jsonl_path: str) -> str:
    # Upload JSONL
    f = client.files.create(
        purpose="batch",
        file=open(jsonl_path, "rb"),
    )
    # Create batch
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return batch.id

def download_batch_output(batch_id: str, out_path: str) -> None:
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch not completed. status={batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but output_file_id is missing.")

    content = client.files.content(batch.output_file_id)
    Path(out_path).write_bytes(content.read())

def parse_batch_output(output_jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            custom_id = obj.get("custom_id")
            response = obj.get("response", {})
            body = response.get("body", {})

            # The structured output is available in the response output; exact field can vary by SDK version.
            # Most reliably, you can parse the assistant message text as JSON.
            # Here we take the first output text chunk.
            output = body.get("output", [])
            parsed = None

            for item in output:
                if item.get("type") == "message":
                    # message.content can be a list; find output_text entries
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            try:
                                parsed = json.loads(c.get("text", ""))
                            except Exception:
                                parsed = None
                            break
                if parsed is not None:
                    break

            if parsed is None:
                # Keep the raw body for debugging
                rows.append({"custom_id": custom_id, "parse_failed": True, "raw": json.dumps(body)[:2000]})
            else:
                rows.append({"custom_id": custom_id, "parse_failed": False, **parsed})

    return pd.DataFrame(rows)

def main():
    # 1) Submit batch
    batch_id = submit_batch(BATCH_INPUT_JSONL)
    print(f"Submitted batch: {batch_id}")
    print("Re-run this script later once the batch completes, or poll status via client.batches.retrieve(batch_id).")

    # NOTE: Uncomment these lines once completed:
    # download_batch_output(batch_id, BATCH_OUTPUT_JSONL)
    # df = parse_batch_output(BATCH_OUTPUT_JSONL)
    # df.to_csv(FINAL_TABLE_CSV, index=False)
    # print(f"Wrote: {FINAL_TABLE_CSV}")

if __name__ == "__main__":
    main()
