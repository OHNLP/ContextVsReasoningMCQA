import contextlib
import os
from typing import Union

import ray
import regex
from openai import AzureOpenAI
from tqdm import tqdm
from vllm import SamplingParams, LLM
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

import global_values as gv

import gc

import numpy as np
import pandas as pd
import torch.cuda
from pandas import DataFrame


def apply_qa(row) -> str:
    question: str = row['question']
    options: dict = row['options']
    messages = [
        {
            "role": "user",
            "content": question
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
        val = model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6))[0].outputs[0].text
        if 'deepseek' in str.lower(model_name):
            val = regex.split('</think>', val)[1]
            val = regex.sub('</?answer>', '', val).strip()
        return val


if __name__ == '__main__':
    '''Obtains a benchmark set of "correct" answers from the MedQA dataset using the parameter model'''
    tqdm.pandas()
    for model_name in gv.models:
        df: DataFrame = pd.read_csv(f'data/outputs/step_1_mcqa_all_models_correct_sample.csv')
        model: Union[None, LLM, AzureOpenAI] = None
        if model_name.startswith('azure'):
            model = gv.get_pipeline_openai()
        else:
            model = gv.get_pipeline_vllm(model_name)
        df['short_answer_02'] = df.progress_apply(apply_qa, axis=1)
        df.to_csv(f'data/outputs/{model_name}/step_2_short_answer_performance.csv')
        destroy_model_parallel()
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

