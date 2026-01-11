import json
from pathlib import Path
import pandas as pd
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
OPENAI_MODEL = "gpt-5.1"
MODEL = "qwen"

BATCH_INPUT_JSONL = f"qwen_output/feature_interpret_requests_{OPENAI_MODEL}_{MODEL}.jsonl"
BATCH_OUTPUT_JSONL = f"qwen_output/batch_output_{OPENAI_MODEL}_{MODEL}.jsonl"
FINAL_TABLE_CSV = f"qwen_output/sae_feature_table_with_interpretations_{OPENAI_MODEL}_{MODEL}.csv"

# Initialize OpenAI client
client = OpenAI()


def submit_batch(jsonl_path: str) -> str:
    """Upload JSONL and create batch job."""
    print(f"Uploading {jsonl_path}...")
    f = client.files.create(
        purpose="batch",
        file=open(jsonl_path, "rb"),
    )
    print(f"File uploaded: {f.id}")
    
    # Create batch
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Batch created: {batch.id}")
    return batch.id


def check_batch_status(batch_id: str) -> dict:
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "request_counts": batch.request_counts,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def download_batch_output(batch_id: str, out_path: str) -> None:
    """Download the output file from a completed batch."""
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        raise RuntimeError(f"Batch not completed. status={batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but output_file_id is missing.")

    print(f"Downloading output file {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id)
    Path(out_path).write_bytes(content.read())
    print(f"Saved to {out_path}")


def parse_batch_output(output_jsonl_path: str) -> pd.DataFrame:
    """Parse the batch output JSONL into a DataFrame."""
    rows = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            custom_id = obj.get("custom_id")
            response = obj.get("response", {})
            body = response.get("body", {})

            output = body.get("output", [])
            parsed = None

            for item in output:
                if item.get("type") == "message":
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
                rows.append({
                    "custom_id": custom_id,
                    "parse_failed": True,
                    "raw": json.dumps(body)[:2000]
                })
            else:
                rows.append({
                    "custom_id": custom_id,
                    "parse_failed": False,
                    **parsed
                })

    return pd.DataFrame(rows)


def main():
    import sys
    
    # Check command line args for different modes
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status" and len(sys.argv) > 2:
            # Check status: python run_batch_table_qwen.py status <batch_id>
            batch_id = sys.argv[2]
            status = check_batch_status(batch_id)
            print(json.dumps(status, indent=2, default=str))
            return
            
        elif command == "download" and len(sys.argv) > 2:
            # Download results: python run_batch_table_qwen.py download <batch_id>
            batch_id = sys.argv[2]
            download_batch_output(batch_id, BATCH_OUTPUT_JSONL)
            df = parse_batch_output(BATCH_OUTPUT_JSONL)
            df.to_csv(FINAL_TABLE_CSV, index=False)
            print(f"Wrote: {FINAL_TABLE_CSV}")
            print(f"Total rows: {len(df)}")
            print(f"Parse failures: {df['parse_failed'].sum()}")
            return
            
        elif command == "parse":
            # Just parse existing output: python run_batch_table_qwen.py parse
            df = parse_batch_output(BATCH_OUTPUT_JSONL)
            df.to_csv(FINAL_TABLE_CSV, index=False)
            print(f"Wrote: {FINAL_TABLE_CSV}")
            return
    
    # Default: submit new batch
    print("Submitting new batch...")
    print(f"Input file: {BATCH_INPUT_JSONL}")
    
    if not Path(BATCH_INPUT_JSONL).exists():
        print(f"ERROR: Input file not found: {BATCH_INPUT_JSONL}")
        print("Run prepare_feature_interp_qwen.py first!")
        return
    
    batch_id = submit_batch(BATCH_INPUT_JSONL)
    
    print(f"\n{'='*60}")
    print(f"Batch submitted successfully!")
    print(f"Batch ID: {batch_id}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Check status:   python {sys.argv[0]} status {batch_id}")
    print(f"2. Once completed: python {sys.argv[0]} download {batch_id}")


if __name__ == "__main__":
    main()
