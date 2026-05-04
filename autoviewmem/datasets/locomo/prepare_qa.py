import argparse
import json
from pathlib import Path

from tqdm import tqdm


def prepare_qa(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_list = []
    total_qa_before = 0

    for dialogue in tqdm(data, desc="Extracting QA", unit="dialogue"):
        qa_pairs = dialogue.get("qa", [])
        total_qa_before += len(qa_pairs)

        for qa in qa_pairs:
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            category = str(qa.get("category", "")).strip()

            if category == "5":
                continue
            if question and answer:
                qa_list.append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"Loaded QA items: {total_qa_before}")
    print(f"Saved QA items: {len(qa_list)}")
    print(f"QA file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Flatten LoCoMo QA pairs.")
    parser.add_argument("--input", default="data/locomo/locomo.json", help="Filtered LoCoMo JSON path")
    parser.add_argument("--output", default="data/locomo/locomo_qa.json", help="Flattened QA output path")
    args = parser.parse_args()
    prepare_qa(args.input, args.output)


if __name__ == "__main__":
    main()
