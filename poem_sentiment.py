import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from tenacity import retry, wait_fixed, stop_after_attempt
import openai
import requests

# Load dataset from local files based on seed
def load_local_dataset(seed):
    base_path = os.path.join("poem_data_splits", f"seed_{seed}")
    if not os.path.exists(base_path):
        raise ValueError(f"Seed directory not found: {base_path}")

    train_path = os.path.join(base_path, "train.csv")
    eval_path = os.path.join(base_path, "eval.csv")
    test_path = os.path.join(base_path, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(eval_path) and os.path.exists(test_path)):
        raise ValueError(f"Missing one or more required files in {base_path} (train.csv, eval.csv, test.csv)")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    test_df = pd.read_csv(test_path)

    return train_df, eval_df, test_df

# Define the prompting functions
def zero_shot_prompt(text):
    return f"Classify the sentiment of the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)\n{text}"

def few_shot_prompt(text):
    examples = """
    Here are examples of sentiment classification:
    1. "and that is why, the lonesome day," -> negative
    2. "with pale blue berries. in these peaceful shades--" -> positive
    3. "it flows so long as falls the rain," -> no_impact
    4. "when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals," -> mixed
    """
    return f"{examples}\nClassify the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)\n{text}"

def chain_of_thought_prompt(text):
    examples = """
    Here are examples of sentiment classification and their reasonings:
    1. "and that is why, the lonesome day," -> negative because the use of "lonesome" suggests sadness or isolation, contributing to a negative sentiment.
    2. "with pale blue berries. in these peaceful shades--" -> positive because the description conveys tranquility and beauty, using calming imagery ("peaceful shades"), which evokes a positive emotional response.
    3. "it flows so long as falls the rain," -> no_impact because the text is neutral and descriptive, focusing on a natural process ("flows" and "rain") without emotional or evaluative language that indicates sentiment.
    4. "when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals," -> mixed because the text acknowledges achievements (positive) but lacks admiration or envy (negative), resulting in a mixed sentiment.
    """
    return f"{examples}\nClassify the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed) using reasoning:\n{text}"

def meta_prompt(text):
    return f"Text:{text}\nSolution Structure: 1. Think about and explain the emotional tone of the text 2. Think about and explain the context of the text 3. Based on tone and context, classify the text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)"

# Retry logic for API calls
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}")

@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_chatgpt_model(api_key: str, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0):
    openai.api_key = api_key
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        return completions.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying model: {e}")
        raise


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_flant5_model(api_key, prompt):
    model_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    print(f"apikey: {api_key}")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error querying Flan-T5 model: {e}")
        print(response.json())
    return pred
    # model_url = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    # headers = {}
    # payload = {
    #     "inputs": prompt,
    #     "temperature": 0.0
    # }
    # try:
    #     response = requests.post(model_url, headers=headers, json=payload)
    #     response.raise_for_status()
    #     return response.json()[0]['generated_text'].strip()
    # except Exception as e:
    #     print(f"Error querying Flan-T5 model: {e}")
    #     raise

# Process the dataset and apply prompts
def process_dataset(dataset, output_folder, setting, model_query_function, api_key=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []

    for example in tqdm(dataset):
        text = example["text"]
        
        try:
            if setting == "zero-shot":
                prompt = zero_shot_prompt(text)
            elif setting == "few-shot":
                prompt = few_shot_prompt(text)
            elif setting == "chain-of-thought":
                prompt = chain_of_thought_prompt(text)
            elif setting == "meta":
                prompt = meta_prompt(text)
            else:
                raise ValueError("Invalid setting provided.")

            prediction = model_query_function(api_key, prompt)
        except Exception as e:
            print(f"Skipping due to repeated failures: {e}")
            prediction = "Error"

        results.append({
            "text": text,
            "prompt": prompt,
            "prediction": prediction
        })

    output_path = os.path.join(output_folder, f"predictions_{setting}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Print usage instructions
def print_usage():
    print("""
Usage:
  python poem_sentiment.py --setting <setting> --model <model> [--api <API_KEY>] [--seed <SEED>]

Arguments:
  --setting <setting>   (required) One of: zero-shot, few-shot, chain-of-thought, meta
  --model <model>       (required) One of: gpt, flant5
  --api <API_KEY>       (optional, required only for GPT models) Your OpenAI API key
  --seed <SEED>         (optional) Seed value to load specific local dataset

Example:
  python poem_sentiment.py --setting zero-shot --model gpt --api YOUR_OPENAI_API_KEY
  python poem_sentiment.py --setting chain-of-thought --model flant5 --seed 0
""")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Poem Sentiment Analysis")
    parser.add_argument("--setting", choices=["zero-shot", "few-shot", "chain-of-thought", "meta"], help="The type of prompt to use.")
    parser.add_argument("--model", choices=["gpt", "flant5"], help="The model to use.")
    parser.add_argument("--api", help="The OpenAI API key (only required for GPT models).")
    parser.add_argument("--seed", type=int, help="Seed value to load a specific local dataset.")
    args = parser.parse_args()

    # Validate arguments
    if not args.setting or not args.model:
        print("Error: Missing required arguments.")
        print_usage()
        return

    if args.model == "gpt" and not args.api:
        print("Error: API key required for GPT models.")
        print_usage()
        return

    if args.seed is None:
        args.seed = 42
    try:
        train_df, eval_df, test_df = load_local_dataset(args.seed)
        # for now we use test dataset since we are not training this model
        dataset = test_df.to_dict(orient="records")
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    output_folder = "poem_sentiment_results"

    api_key = args.api
    model_query_function = query_chatgpt_model if args.model == "gpt" else query_flant5_model

    process_dataset(dataset, output_folder, args.setting, model_query_function, api_key)

if __name__ == "__main__":
    main()
