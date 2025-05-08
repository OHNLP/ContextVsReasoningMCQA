import datasets
import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("hf://datasets/GBaker/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl", lines=True)
    df = df[df['meta_info'] == 'step2&3']
    df = df.assign(
    option = df['options']
               .apply(lambda d: list(d.values()))
    ).explode('option')['option']
    df.to_csv('data/input/expanded_context_opt_medical.txt', index=False, header=False)