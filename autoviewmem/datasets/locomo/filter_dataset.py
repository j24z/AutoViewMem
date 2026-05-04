import argparse
import json
from pathlib import Path


def filter_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for block in data:
        if "qa" in block and isinstance(block["qa"], list):
            block["qa"] = [
                item for item in block["qa"]
                if not (isinstance(item, dict) and str(item.get("category")) == "5")
            ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Filtered dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove LoCoMo category-5 QA items.")
    parser.add_argument("--input", required=True, help="Raw LoCoMo JSON path")
    parser.add_argument("--output", default="data/locomo/locomo.json", help="Filtered output path")
    args = parser.parse_args()
    filter_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
