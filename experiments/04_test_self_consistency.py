import ast
import contextlib
import gc
import traceback
from typing import Union

import pandas as pd
import ray
import torch
from openai import AzureOpenAI
from pandas import DataFrame
from regex import regex
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

import global_values as gv


def apply_permute_answer(row):
    question: str = row['question']
    options: dict = ast.literal_eval(row['options'])
    answer: str = row['answer_idx']

    ret = []
    for k, v in options.items():
        output_row = row.copy()
        prompt = (f"Determine whether the provided answer is correct for the question. Answer with only \"YES\" or \"NO\".\n\n"
                  f"The question:\n{question}\n\nThe answer:\n{v}")
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        output_row['permuted_answer'] = v
        for i in range(0, 5):
            model_answer = None
            if model_name.startswith('azure'):
                try:
                    val = model.chat.completions.create(
                        model=model_name[6:],
                        messages=messages,
                        max_completion_tokens=10000
                    )
                    model_answer = val.choices[0].message.content.strip()
                except Exception as e:
                    print(e)
                    model_answer = ''
            else:
                to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True,
                                                                     tokenize=False)
                model_answer = \
                model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6), use_tqdm=False)[0].outputs[
                    0].text
                if 'deepseek' in str.lower(model_name) or 'qwq' in str.lower(model_name):
                    try:
                        model_answer = regex.split('</think>', model_answer)[1]
                        model_answer = regex.sub('</?answer>', '', model_answer).strip()
                    except Exception as e:
                        print('keeping original model answer due to CoT parsing failure\r\n')
                        print(model_answer)
                        traceback.print_exc()
            output_row[f'permuted_model_answer_{i}'] = model_answer
        ret.append(output_row)
        # output_row['permuted_model_answer_correct'] = model_answer.upper() == 'NO' if k != answer else model_answer.upper() == 'YES'
        # if model_answer.upper() == 'NO' and k != answer:
        #     prompt = (
        #         f"Determine whether the provided answer is correct for the question. If yes, respond only with \"CORRECT\", if not, respond with the answer.\n\n"
        #         f"The question:\n{question}\n\nThe answer:\n{v}")
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        #     model_answer = None
        #     if model_name.startswith('azure'):
        #         try:
        #             val = model.chat.completions.create(
        #                 model=model_name[6:],
        #                 messages=messages,
        #                 max_completion_tokens=2000
        #             )
        #             model_answer = val.choices[0].message.content.strip()
        #         except Exception as e:
        #             print(e)
        #             model_answer = ''
        #     else:
        #         to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True,
        #                                                              tokenize=False)
        #         model_answer = \
        #         model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6))[0].outputs[
        #             0].text
        #         if 'deepseek' in str.lower(model_name):
        #             model_answer = regex.split('</think>', model_answer)[1]
        #             model_answer = regex.sub('</?answer>', '', model_answer).strip()
        #     output_row['permuted_model_corrected_answer'] = model_answer
        # else:
        #     output_row['permuted_model_corrected_answer'] = None
        # ret.append(output_row)
    return ret

if __name__ == '__main__':
    '''
    Substitutes in incorrect answers from MCQA prompt, presenting it as the correct answer, and evaluates whether the
    LLM is capable of identifying an incorrect answer as such. If/when it does identify that this is incorrect, prompts
    the LLM to further specify the correct answer (short answer). 
    '''
    tqdm.pandas()
    for model_name in gv.models:
        model: Union[None, LLM, AzureOpenAI] = None
        if model_name.startswith('azure'):
            model = gv.get_pipeline_openai()
        else:
            model = gv.get_pipeline_vllm(model_name)
        for dataset_name in gv.datasets:
            df: DataFrame = pd.read_csv(f'data/outputs/{dataset_name}/step_1_mcqa_all_models_correct_sample.csv')
            df= df.progress_apply(apply_permute_answer, axis=1)
            df = pd.DataFrame(df.explode().tolist())
            df.to_csv(f'data/outputs/{dataset_name}/{model_name}/step_4_self_consistency_permuted.csv')

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





