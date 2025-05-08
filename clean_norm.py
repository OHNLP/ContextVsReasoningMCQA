import os.path
import re
from typing import List

import pandas as pd
from pandas import DataFrame


def clean(input: str):
    split = input.split('\n')
    # First check last line
    last_line = split[len(split)-1].strip()
    if len(last_line) == 1 and re.search('[A-Z]', last_line.upper()):
        return last_line
    elif len(last_line) == 2 and last_line[1] == '.' and re.search('[A-Z]', last_line[0].upper()):
        return last_line[0]
    else:
        match = re.search('^([A-Z]):', last_line.upper())
        if match:
            return match.group(1)
        match = re.search('ANSWER:\\s+?([A-Z])', last_line.upper())
        if match:
            return match.group(1)

    # Now check first line
    first_line = split[0].strip()
    if len(first_line) == 1:
        return first_line
    elif len(first_line) == 2 and first_line[1] == '.':
        return first_line[0]
    else:
        match = re.search('^([A-Z]):', first_line.upper())
        if match:
            return match.group(1)
        match = re.search('ANSWER:\\s+?([A-Z])', first_line.upper())
        if match:
            return match.group(1)
        # match = re.search('THE (?:MOST )?(?:LIKELY|CORRECT|APPROPRIATE)[^.]+(?:IS|WOULD BE):?(?: OPTION)?:? ([A-E])', first_line.upper())
        # if match:
        #     return match.group(1)
        #
        # match = re.search('ASSISTANT:\\s*([A-E])', first_line.upper())
        # if match:
        #     return match.group(1)
        if len(input) > 32767: # Excel cell char limit
            return input[0:32500]
        return input

if __name__ == '__main__':
    datasets: List[str] = [
        "medqa",
        "mmlu",
        "jama",
        "medbullets"
    ]

    models: List[str] = [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "azure/o1",
        "Qwen/QwQ-32B",
        "Qwen/Qwen2.5-32B-Instruct"
    ]
    working_dir = ''  # diffs/Qwen/Qwen2.5-32B-Instruct or blank
    for model in models:
        for dataset in datasets:
            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_6_self_consistency_permuted_new_normed.csv'):
                if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_6_self_consistency_permuted_new.csv'):
                    print(f'Skipping experiment 1 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df: DataFrame = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_6_self_consistency_permuted_new.csv', index_col='question_idx')
                for i in range(0, 5):
                    df[f'permuted_model_answer_{i}'] = df[f'permuted_model_answer_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_6_self_consistency_permuted_new_normed.csv')
            else:
                print(f'Skipping experiment 1 for {model} on dataset {dataset} due to norm already complete')

            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_5_no_correct_answer_normed.csv'):
                if not os.path.exists(
                    f'data/outputs/{dataset}/{working_dir}/{model}/step_5_no_correct_answer.csv'):
                    print(f'Skipping  experiment 2/5 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_5_no_correct_answer.csv', index_col='question_idx')
                # df = df.drop(df.columns[0], axis=1)
                for i in range(0, 5):
                    df[f'no_valid_option_nonexplicit_{i}'] = df[f'no_valid_option_nonexplicit_{i}'].map(clean)
                for i in range(0, 5):
                    df[f'no_valid_option_explicit_{i}'] = df[f'no_valid_option_explicit_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_5_no_correct_answer_normed.csv')
            else:
                print(f'Skipping experiment 2/5 for {model} on dataset {dataset} due to norm already complete')

            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_8_long_context_normed.csv'):
                if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_8_long_context.csv'):
                    print(f'Skipping experiment 3 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df: DataFrame = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_8_long_context.csv', index_col='question_idx')
                for i in range(0, 5):
                    df[f'expanded_context_model_answer_{i}'] = df[f'expanded_context_model_answer_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_8_long_context_normed.csv')
            else:
                print(f'Skipping experiment 3 for {model} on dataset {dataset} due to norm already complete')

            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_7_self_consistency_permuted_no_context_normed.csv'):
                if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_7_self_consistency_permuted_no_context.csv'):
                    print(f'Skipping experiment 4 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df: DataFrame = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_7_self_consistency_permuted_no_context.csv', index_col='question_idx')
                for i in range(0, 5):
                    df[f'permuted_model_answer_{i}'] = df[f'permuted_model_answer_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_7_self_consistency_permuted_no_context_normed.csv')
            else:
                print(f'Skipping experiment 4 for {model} on dataset {dataset} due to norm already complete')

            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_9_shuffled_context_normed.csv'):
                if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_9_shuffled_context.csv'):
                    print(f'Skipping experiment 6 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df: DataFrame = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_9_shuffled_context.csv', index_col='question_idx')
                for i in range(0, 5):
                    df[f'shuffled_context_model_answer_{i}'] = df[f'shuffled_context_model_answer_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_9_shuffled_context_normed.csv')
            else:
                print(f'Skipping experiment 6 for {model} on dataset {dataset} due to norm already complete')
            if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_10_long_context_med_normed.csv'):
                if not os.path.exists(f'data/outputs/{dataset}/{working_dir}/{model}/step_10_long_context_med.csv'):
                    print(f'Skipping experiment 3 for {model} on dataset {dataset} due to no data being run yet')
                    continue
                df: DataFrame = pd.read_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_10_long_context_med.csv', index_col='question_idx')
                for i in range(0, 5):
                    df[f'expanded_context_model_answer_{i}'] = df[f'expanded_context_model_answer_{i}'].map(clean)
                df.to_csv(f'data/outputs/{dataset}/{working_dir}/{model}/step_10_long_context_med_normed.csv')
            else:
                print(f'Skipping experiment 3 for {model} on dataset {dataset} due to norm already complete')