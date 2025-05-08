import ast
import os
import random
from typing import Union

import regex
from openai import AzureOpenAI
from tqdm import tqdm
from vllm import SamplingParams, LLM
from vllm.distributed import destroy_model_parallel

import global_values as gv

import gc

import pandas as pd
import torch.cuda
from pandas import DataFrame


def apply_qa(row) -> str:
    question: str = row['question']
    options: dict = ast.literal_eval(row['options'])
    prompt: str = question + "\n Options: \n"
    drug_options = []
    for k, v in options.items():
        drug_options.append(v)
    for item in all_drugs:
        drug_options.append(item)
    random.shuffle(drug_options)
    for option in drug_options:
        prompt = prompt + '- ' + option + '\n'
    messages = [
        {
            "role": "user",
            "content": "Select the appropriate option from the provided question. "
                       "Respond with the exact answer as presented in the options \n\n"
                       "The question:\n" + prompt
        }
    ]
    if model_name.startswith('azure'):
        try:
            val = model.chat.completions.create(
                model=model_name[6:],
                messages=messages,
                max_completion_tokens=3000
            )
            ret = val.choices[0].message.content.strip()
            return ret
        except Exception as e:
            print(e)
            return ''
    else:
        to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        val = model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6))[0].outputs[
            0].text
        if val.startswith('<think>'):
            val = regex.split('</?think>', val)[2]
        if val.startswith('<answer>'):
            val = regex.sub('</?answer>', '', val)
        return val


if __name__ == '__main__':
    '''Obtains a benchmark set of "correct" answers from the MedQA dataset using the parameter model'''
    all_drugs = []
    with open('data/input/drug_names.txt', 'r') as f:
        for line in f:
            all_drugs.append(line.rstrip())
    tqdm.pandas()
    for model_name in gv.models:
        df: DataFrame = pd.read_csv(f'data/outputs/step_1_mcqa_all_models_correct_sample_drugs_only.csv')
        model: Union[None, LLM, AzureOpenAI] = None
        if model_name.startswith('azure'):
            model = gv.get_pipeline_openai()
        else:
            model = gv.get_pipeline_vllm(model_name)
        df['context_expansion_answer'] = df.progress_apply(apply_qa, axis=1)
        df['context_expansion_answer_correct'] = df['context_expansion_answer'] == df['answer']
        df.to_csv(f'data/outputs/{model_name}/step_3_short_long_context.csv')
        if not model_name.startswith('azure'):
            destroy_model_parallel()
            del model.llm_engine.driver_worker
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
