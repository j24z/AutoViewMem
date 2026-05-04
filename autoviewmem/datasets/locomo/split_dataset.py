import argparse
import json
from pathlib import Path


def split_dataset(input_path, output_dir, train_size=1, val_size=1):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    files = {
        "train": output_dir / "locomo_train.json",
        "val": output_dir / "locomo_val.json",
        "test": output_dir / "locomo_test.json",
    }

    for split, path in files.items():
        payload = {"train": train_data, "val": val_data, "test": test_data}[split]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Train conversations: {len(train_data)} -> {files['train']}")
    print(f"Validation conversations: {len(val_data)} -> {files['val']}")
    print(f"Test conversations: {len(test_data)} -> {files['test']}")


def main():
    parser = argparse.ArgumentParser(description="Split LoCoMo conversations into train/validation/test files.")
    parser.add_argument("--input", default="data/locomo/locomo.json", help="Filtered LoCoMo JSON path")
    parser.add_argument("--output_dir", default="data/locomo", help="Directory for split files")
    parser.add_argument("--train_size", type=int, default=1, help="Number of conversations in train split")
    parser.add_argument("--val_size", type=int, default=1, help="Number of conversations in validation split")
    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, args.train_size, args.val_size)


if __name__ == "__main__":
    main()
