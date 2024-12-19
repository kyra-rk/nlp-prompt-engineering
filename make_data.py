import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_poem_sentiment_dataset():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    return dataset  # Now returning the entire dataset (train, validation, test)

def create_data(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the entire dataset (train, validation, test)
    dataset = load_poem_sentiment_dataset()

    # Combine the 'train', 'validation', and 'test' splits into one dataframe
    data = pd.concat([
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["train"]], "label": [example["label"] for example in dataset["train"]]}),
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["validation"]], "label": [example["label"] for example in dataset["validation"]]}),
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["test"]], "label": [example["label"] for example in dataset["test"]]})
    ], ignore_index=True)

    data.to_csv(os.path.join(output_dir, "poem_sentiment_data.csv"), index=False)

def print_usage():
    print("""
Usage:
  python make_data.py --output_dir <output_directory>

Arguments:
  --output_dir <output_directory>   (required) Directory to save the data

Example:
  python make_data.py --output_dir poem_data
""")

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for the Poem Sentiment dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the data splits.")
    args = parser.parse_args()

    if not args.output_dir:
        print("Error: --output_dir argument is required.")
        print_usage()
        return

    create_data(args.output_dir)

if __name__ == "__main__":
    main()
