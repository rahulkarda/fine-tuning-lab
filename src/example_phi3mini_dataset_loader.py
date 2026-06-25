from src.utils import load_jsonl, dataset_stats

"""
Minimal stub for loading and inspecting Phi-3-mini instruction dataset.
For phase 4 experiment: phi-3-mini on instruction data.

Replace 'data/phi3mini_instruct.jsonl' with your actual dataset path.
Run directly to print basic stats.
"""

if __name__ == "__main__":
    dataset_path = "data/phi3mini_instruct.jsonl"  # placeholder path
    try:
        data = load_jsonl(dataset_path)
        print(f"Loaded {len(data)} examples from {dataset_path}")
        stats = dataset_stats(data)
        print("--- Dataset Stats ---")
        for k, v in stats.items():
            print(f"{k}: {v}")
    except FileNotFoundError:
        print(f"Dataset not found: {dataset_path}. Please check the path.")
