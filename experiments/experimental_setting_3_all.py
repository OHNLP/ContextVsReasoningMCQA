import ast
import multiprocessing
import os
from typing import Union

import pandas as pd
import regex
from openai import AzureOpenAI
from pandas import DataFrame
from tqdm import tqdm
from vllm import SamplingParams, LLM

import global_values as gv

opt_array = ['A', 'B', 'C', 'D', 'E']


def apply_qa_nonexplicit(row, model=None, model_path=None) -> str:
    question: str = row['question']
    options: dict = ast.literal_eval(row['options'])
    answer: str = row['answer_idx']
    prompt: str = question + "\n Options: \n"
    num_opts_prefix = len(options.items()) - 2
    response_format = ', '.join(opt_array[0:num_opts_prefix])
    response_format = response_format + f", or {opt_array[num_opts_prefix]}"
    i = 0
    for k, v in options.items():
        if k == answer:
            continue
        prompt = prompt + opt_array[i] + ': ' + v + '\n'
        i += 1
    messages = [
        {
            "role": "user",
            "content": f"Select the appropriate option from the provided question."
                       f"Respond in the format {response_format}. If no appropriate option is provided, respond with {opt_array[num_opts_prefix + 1]} \n\n"
                       f"The question:\n" + prompt
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
            except:
                return val
        return val


def apply_qa_explicit(row, model=None, model_path=None) -> str:
    question: str = row['question']
    options: dict = ast.literal_eval(row['options'])
    answer: str = row['answer_idx']
    prompt: str = question + "\n Options: \n"
    num_opts_index = len(options.items()) - 1
    response_format = ', '.join(opt_array[0:num_opts_index])
    response_format = response_format + f", or {opt_array[num_opts_index]}"
    i = 0
    for k, v in options.items():
        if k == answer:
            continue
        prompt = prompt + opt_array[i] + ': ' + v + '\n'
        i += 1
    prompt = prompt + f'{opt_array[num_opts_index]}: None of the above\n'
    messages = [
        {
            "role": "user",
            "content": f"Select the appropriate option from the provided question. "
                       f"Respond in the format {response_format}. \n\n"
                       f"The question:\n" + prompt
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
            except:
                return val
        return val


def call_experimental_setting_3_all(model_name, dataset_name, working_dir):
    tqdm.pandas()
    model: Union[None, LLM, AzureOpenAI] = None
    if model_name.startswith('azure'):
        model = gv.get_pipeline_openai()
    else:
        model = gv.get_pipeline_vllm(model_name)
    df: DataFrame = pd.read_csv(f'data/outputs/{dataset_name}/{working_dir}/step_1_mcqa_all_models_correct_sample.csv',
                                index_col='question_idx')
    for i in range(0, 5):
        df[f'no_valid_option_nonexplicit_{i}'] = df.progress_apply(lambda r: apply_qa_nonexplicit(r, model, model_name), axis=1)
    df['no_valid_option_nonexplicit_answer'] = None
    df['no_valid_option_nonexplicit_answer_correct'] = None
    for i in range(0, 5):
        df[f'no_valid_option_explicit_{i}'] = df.progress_apply(lambda r: apply_qa_explicit(r, model, model_name), axis=1)
    df['no_valid_option_explicit_answer'] = None
    df['no_valid_option_explicit_answer_correct'] = None

    output_dir = f'data/outputs/{dataset_name}/{working_dir}/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(f'data/outputs/{dataset_name}/{working_dir}/{model_name}/step_5_no_correct_answer.csv')


if __name__ == '__main__':
    '''Obtains a benchmark set of "correct" answers from the MedQA dataset using the parameter model'''
    for mn in gv.models:
        for dn in gv.datasets:
            process = multiprocessing.Process(target=call_experimental_setting_3_all,
                                              args=(mn, dn, 'diffs/Qwen/Qwen2.5-32B-Instruct'), name="exp_setting_2_5")
            process.start()
            process.join()
