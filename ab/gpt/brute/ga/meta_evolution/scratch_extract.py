import json
import sys

log_file = "logs/LLM-evolution-logs_2026-06-19_16-53-28.jsonl"
print("Parsing", log_file)
with open(log_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("valid_syntax"):
                print("="*50)
                print(f"Timestamp: {data.get('timestamp')}")
                print(f"Method evolved: {data.get('method')}")
                print(f"Attempt: {data.get('attempt')}")
                print(f"Peak Accuracy: {data.get('peak_accuracy', data.get('score'))}")
                print(f"Code Snippet:\n{data.get('cleaned_code', '')[:300]}...")
        except Exception as e:
            pass
