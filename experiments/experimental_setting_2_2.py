import ast
import multiprocessing
import os
import traceback

import pandas as pd
from pandas import DataFrame
from regex import regex
from tqdm import tqdm
from vllm import SamplingParams

import global_values as gv


def apply_permute_answer(row, model=None, model_path=None):
    question: str = row['question']
    options: dict = ast.literal_eval(row['options'])
    answer: str = row['answer_idx']

    ret = []
    for k, v in options.items():
        other_opts = []
        for k1, v1 in options.items():
            if k1 == k:
                continue
            else:
                other_opts.append(v1)
        output_row = row.copy()
        output_row.name = row.name
        prompt = (f"Select the appropriate option from the provided question.\n Respond in the format A or B.\n\n"
                  f"The question:\n{question}\n Is the answer {v} correct?\n\n"
                  f"Options:\nA: Yes\nB: No")
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        output_row['permuted_answer'] = v
        for i in range(0, 5):
            model_answer = None
            if model_path.startswith('azure'):
                try:
                    val = model.chat.completions.create(
                        model=model_path[6:],
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
                if 'deepseek' in str.lower(model_path) or 'qwq' in str.lower(model_path):
                    try:
                        model_answer = regex.split('</think>', model_answer)[1]
                        model_answer = regex.sub('</?answer>', '', model_answer).strip()
                    except Exception as e:
                        print('keeping original model answer due to CoT parsing failure\r\n')
                        print(model_answer)
                        traceback.print_exc()
            output_row[f'permuted_model_answer_{i}'] = model_answer
        output_row['permuted_model_answer'] = ''
        output_row['permuted_correct_answer'] = 'A' if k == answer else 'B'
        output_row['permuted_answer_correct'] = ''
        ret.append(output_row)

    return ret


def call_experimental_setting_2_2(model_name, dataset_name, working_dir):
    tqdm.pandas()
    if model_name.startswith('azure'):
        model = gv.get_pipeline_openai()
    else:
        model = gv.get_pipeline_vllm(model_name)
    df: DataFrame = pd.read_csv(f'data/outputs/{dataset_name}/{working_dir}/step_1_mcqa_all_models_correct_sample.csv',
                                index_col='question_idx')
    df = df.progress_apply(lambda r: apply_permute_answer(r, model, model_name), axis=1)
    df = pd.DataFrame(df.explode().tolist())
    output_dir = f'data/outputs/{dataset_name}/{working_dir}/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(f'data/outputs/{dataset_name}/{working_dir}/{model_name}/step_7_self_consistency_permuted_no_context.csv')


if __name__ == '__main__':
    '''
    Substitutes in incorrect answers from MCQA prompt, presenting it as the correct answer, and evaluates whether the
    LLM is capable of identifying an incorrect answer as such. If/when it does identify that this is incorrect, prompts
    the LLM to further specify the correct answer (short answer). 
    '''
    for mn in gv.models:
        for dn in gv.datasets:
            process = multiprocessing.Process(target=call_experimental_setting_2_2,
                                              args=(mn, dn, ''),
                                              name="exp_setting_4")
            process.start()
            process.join()




