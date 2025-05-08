import ast
import multiprocessing
import os
import random
from typing import Union

import pandas as pd
import regex
from openai import AzureOpenAI
from pandas import DataFrame
from tqdm import tqdm
from vllm import SamplingParams, LLM

import global_values as gv

with open('data/input/expanded_context_opt_medical.txt', 'r') as file:
    master_additional_option_list = [line.rstrip('\n') for line in file.readlines()]


def expand_and_shuffle_options(row):
    options: dict = ast.literal_eval(row['options'])
    output_options = []
    for k, v in options.items():
        output_options.append(v)
    # We must shuffle here first to ensure that we don't add the same set of additional options every time
    shuffled_additional_options = master_additional_option_list.copy()
    random.shuffle(shuffled_additional_options)
    accession_idx = 0
    while len(output_options) < 26:
        output_options.append(shuffled_additional_options[accession_idx])
        accession_idx += 1
    # Now shuffle output options
    random.shuffle(output_options)
    # Reconstruct the dictionary
    answer_idx = None
    output_option_dict = {}
    offset_idx = 0
    answer = row['answer']
    for v in output_options:
        char_idx = chr(ord('A') + offset_idx)
        output_option_dict[char_idx] = v
        if v == answer:
            answer_idx = char_idx
        offset_idx += 1
    # Now construct the output row
    output_row = row.copy()
    output_row['expanded_context_options'] = output_option_dict
    output_row['expanded_context_answer_idx'] = answer_idx
    return output_row


def apply_qa(row, model=None, model_path=None) -> str:
    question: str = row['question']
    options: dict = row['expanded_context_options']
    prompt: str = question + "\n Options: \n"
    for k, v in options.items():
        prompt = prompt + k + ': ' + v + '\n'
    messages = [
        {
            "role": "user",
            "content": "Select the appropriate option from the provided question. "
                       "Respond in the format A, B, C, ..., Z \n\n"
                       "The question:\n" + prompt
        }
    ]
    if model_path.startswith('azure'):
        try:
            val = model.chat.completions.create(
                model=model_path[6:],
                messages=messages,
                max_completion_tokens=10000
            )
            ret = val.choices[0].message.content.strip()
            return ret
        except Exception as e:
            print(e)
            return ''
    else:
        to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        val = \
        model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6), use_tqdm=False)[
            0].outputs[0].text
        if 'deepseek' in str.lower(model_path) or 'qwq' in str.lower(model_path):
            try:
                val = regex.split('</think>', val)[1]
                val = regex.sub('</?answer>', '', val).strip()
            except Exception as e:
                return val
        return val


def call_experimental_setting_1_2_2(model_name, dataset_name, working_dir):
    tqdm.pandas()
    model: Union[None, LLM, AzureOpenAI] = None
    if model_name.startswith('azure'):
        model = gv.get_pipeline_openai()
    else:
        model = gv.get_pipeline_vllm(model_name)
    df: DataFrame = pd.read_csv(
        f'data/outputs/{dataset_name}/{working_dir}/step_1_mcqa_all_models_correct_sample.csv',
        index_col='question_idx')
    df = df.progress_apply(expand_and_shuffle_options, axis=1)
    for i in range(0, 5):
        df[f'expanded_context_model_answer_{i}'] = df.progress_apply(lambda r: apply_qa(r, model, model_name), axis=1)
    df['expanded_context_model_answer'] = ''
    df['expanded_context_answer_correct'] = ''
    output_dir = f'data/outputs/{dataset_name}/{working_dir}/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(f'data/outputs/{dataset_name}/{working_dir}/{model_name}/step_10_long_context_med.csv')

if __name__ == '__main__':
    for mn in gv.models:
        for dn in gv.datasets:
            process = multiprocessing.Process(target=call_experimental_setting_1_2_2,
                                              args=(mn, dn, ''),
                                              name="exp_setting_1_2_2")
            process.start()
            process.join()