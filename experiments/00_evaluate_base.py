import contextlib
import gc
import os
from typing import Union

import numpy as np
import pandas as pd
import ray
import regex
import torch.cuda
from openai import AzureOpenAI
from pandas import DataFrame
from tqdm import tqdm
from vllm import SamplingParams, LLM
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

import global_values as gv
from global_values import HuggingfaceModelWrapper


def apply_qa(row) -> str:
    question: str = row['question']
    options: dict = row['options']
    prompt: str = question + "\n Options: \n"
    for k, v in options.items():
        prompt = prompt + k + ': ' + v + '\n'
    messages = [
        {
            "role": "user",
            "content": "Select the appropriate option from the provided question. "
                       "Respond in the format A, B, C, D, or E. \n\n"
                       "The question:\n" + prompt
        }
    ]
    if model_name.startswith('azure'):
        try:
            val = model.chat.completions.create(
                model=model_name[6:],
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
        val = model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6), use_tqdm=False)[0].outputs[0].text
        if 'deepseek' in str.lower(model_name) or 'qwq' in str.lower(model_name):
            try:
                val = regex.split('</think>', val)[1]
                val = regex.sub('</?answer>', '', val).strip()
            except Exception as e:
                return val
        return val

def map_option_list_to_dict_mmlu(row):
    opt = ['A', 'B', 'C', 'D', 'E']
    option_list = row['options_list']
    ret = {}
    for i in range(0, min(len(opt), len(option_list))):
        ret[opt[i]] = option_list[i]

    return ret

def map_options_medbullets(row):
    opt = ['A', 'B', 'C', 'D', 'E']
    ret = {}
    for option_idx in opt:
        ret[option_idx] = row[f"choices{option_idx}"]

    return ret

def map_options_jama(row):
    opt = ['A', 'B', 'C', 'D', 'E']
    ret = {}
    for option_idx in opt:
        if f"op{option_idx.lower()}" in row:
            ret[option_idx] = row[f"op{option_idx.lower()}"]

    return ret


def load_dataset(dataset_name: str) -> DataFrame:
    data_file_name = f'data/input/{dataset_name}.jsonl'
    # # Special handling for JAMA import as the question bank is too large for 5x inference on our hardware
    # if dataset_name == 'jama':
    #     if not os.path.exists(f'data/input/{dataset_name}_sampled.jsonl'):
    #         with open(data_file_name, 'r') as f:
    #             df: DataFrame = pd.read_json(f, lines=True)
    #             df['question_idx'] = np.arange(df.shape[0])
    #             df.sample(500).to_json(f'data/input/{dataset_name}_sampled.jsonl', lines=True)
    #     data_file_name = f'data/input/{dataset_name}_sampled.jsonl'

    with open(data_file_name, 'r') as f:
        df: DataFrame = pd.read_json(f, lines=True)
        # if dataset_name != 'jama':
        # # JAMA question indexes already handled during sampling.
        # This should be wholly reproducible as the order the questions are loaded from jsonl do not change and we do not shuffle here
        df['question_idx'] = np.arange(df.shape[0])
    if dataset_name == 'medqa':
        # Restrict to step 2&3 (4_options version as is reported in MedQA paper)
        df = df[df['meta_info'] == 'step2&3']
    elif dataset_name == 'mmlu':
        df['question'] = df['centerpiece']
        df['answer'] = df['correct_options_literal'].str[0]
        df.rename(columns={'options': 'options_list'}, inplace=True)
        df['options'] = df.apply(map_option_list_to_dict_mmlu, axis=1)
        df['answer_idx'] = df['correct_options'].str[0]
        df = df[['question_idx', 'question', 'answer', 'options', 'answer_idx']]
    elif dataset_name == 'medbullets':
        df['options'] = df.apply(map_options_medbullets, axis=1)
        df = df[['question_idx', 'question', 'answer', 'options', 'answer_idx']]
    elif dataset_name == 'jama':
        df['options'] = df.apply(map_options_jama, axis=1)
        df = df[['question_idx', 'question', 'answer', 'options', 'answer_idx']]

    return df


if __name__ == '__main__':
    '''Obtains a benchmark set of "correct" answers from the MedQA dataset using the parameter model'''
    tqdm.pandas()
    for model_name in gv.models:
        model: Union[None, LLM, AzureOpenAI, HuggingfaceModelWrapper] = None
        if model_name.startswith('azure'):
            model = gv.get_pipeline_openai()
        elif model_name.startswith('opea'):
            model = gv.get_pipeline_deepseek(model_name)
        else:
            model = gv.get_pipeline_vllm(model_name)
        for dataset_name in gv.datasets:
            df = load_dataset(dataset_name)
            os.makedirs(f'data/outputs/{dataset_name}/{model_name}', exist_ok=True)
            df['base_answer_0'] = df.progress_apply(apply_qa, axis=1)
            df['base_answer_1'] = df.progress_apply(apply_qa, axis=1)
            df['base_answer_2'] = df.progress_apply(apply_qa, axis=1)
            df['base_answer_3'] = df.progress_apply(apply_qa, axis=1)
            df['base_answer_4'] = df.progress_apply(apply_qa, axis=1)
            df.to_csv(f'data/outputs/{dataset_name}/{model_name}/step_0_raw_mcqa_performance.csv')
        if not model_name.startswith('azure'):
            destroy_model_parallel()
            destroy_distributed_environment()
            del model.llm_engine.model_executor
            del model
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
