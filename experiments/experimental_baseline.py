import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import regex
from pandas import DataFrame
from tqdm import tqdm
from vllm import SamplingParams

import global_values as gv


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

def build_prompt(question, options):
    prompt: str = (f"Select the appropriate option from the provided question. "
                   f"Respond in the format A, B, C, D, or E. \n\n"
                   f"The question:\n{question}"
                   f"\n Options: \n")
    for k, v in options.items():
        prompt = prompt + k + ': ' + v + '\n'
    return prompt


def call_baseline(model_name, dataset_name, working_dir, gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    model = (gv.get_pipeline_openai() if model_name.startswith('azure')
             else gv.get_pipeline_vllm(model_name))
    df = load_dataset(dataset_name)

    answers = {i: [] for i in range(5)}

    with tqdm(total=len(df), desc=f"{model_name}/{dataset_name}", position=gpu_ids[0], leave=True) as bar:
        for row in df.itertuples(index=False):
            prompt = build_prompt(row.question, row.options)
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            if model_name.startswith('azure'):
                try:
                    val = model.chat.completions.create(
                        model=model_name[6:],
                        messages=messages,
                        max_completion_tokens=10000,
                        n=5
                    )
                    ret = [c.message.content.strip() for c in val.choices]
                except Exception as e:
                    print(e)
                    ret = ['', '', '', '', '']
            else:
                to_model = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True,
                                                                     tokenize=False)

                val = \
                    model.generate(to_model, sampling_params=SamplingParams(max_tokens=10000, temperature=0.6, n=5),
                                   use_tqdm=False)[0].outputs
                ret = [o.text.strip() for o in val]
                if 'deepseek' in str.lower(model_name) or 'qwq' in str.lower(model_name):
                    for i in range(0, len(ret)):
                        try:
                            txt = val[i]
                            txt = regex.split('</think>', txt)[1]
                            txt = regex.sub('</?answer>', '', txt).strip()
                            ret[i] = txt.strip()
                        except Exception as e:
                            pass

                ret = [o.strip() for o in ret]

            for i, ans in enumerate(ret):
                answers[i].append(ans)
            bar.update(1)

    for i in range(5):
        df[f"base_answer_{i}"] = answers[i]
    os.makedirs(f"data/outputs/{dataset_name}/{model_name}", exist_ok=True)
    df.to_csv(f"data/outputs/{dataset_name}/{model_name}/step_0_raw_mcqa_performance.csv",
              index=False)

if __name__ == '__main__':
    num_gpus_used = 0
    max_num_gpus = 8
    jobs = [
        (mn, dn, gv.tens_parallelism_setting[mn])
        for mn in gv.models
        for dn in gv.datasets
        if not os.path.exists(f'data/outputs/{dn}/{mn}/step_0_raw_mcqa_performance.csv')
    ]
    active = []  # list of dicts: {'proc': p, 'gpus': gpu_ids, 'req': req}

    while jobs or active:
        for entry in active[:]:
            p = entry['proc']
            if p.exitcode is not None:  # done
                num_gpus_used -= entry['req']
                print(f"Done {entry['gpus']}, freed GPUs ({num_gpus_used}/{max_num_gpus} in use)")
                active.remove(entry)

        scheduled = False
        for (mn, dn, req) in jobs[:]:
            if num_gpus_used + req <= max_num_gpus:
                start = num_gpus_used
                gpu_ids = list(range(start, start + req))
                num_gpus_used += req

                print(f"Launching {dn} on {mn} with GPUs {gpu_ids} ({num_gpus_used}/{max_num_gpus})")
                p = multiprocessing.Process(
                    target=call_baseline,
                    args=(mn, dn, '', gpu_ids),
                    name=f"Baseline-{mn}-{dn}"
                )
                p.start()
                active.append({'proc': p, 'gpus': gpu_ids, 'req': req})
                jobs.remove((mn, dn, req))
                scheduled = True
        if not scheduled:
            time.sleep(10)

    print("All baselines complete.")
