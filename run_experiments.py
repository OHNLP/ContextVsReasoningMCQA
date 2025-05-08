import multiprocessing
import os

import global_values
from experiments.experimental_setting_1_1 import call_experimental_setting_1_1
from experiments.experimental_setting_1_2_1 import call_experimental_setting_1_2_1
from experiments.experimental_setting_1_2_2 import call_experimental_setting_1_2_2

from experiments.experimental_setting_2_1 import call_experimental_setting_2_1
from experiments.experimental_setting_2_2 import call_experimental_setting_2_2
from experiments.experimental_setting_3_all import call_experimental_setting_3_all

if __name__ == '__main__':
    for dataset_name in global_values.datasets:
        for model_name in global_values.models:
            # -- Setting 1
            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_9_shuffled_context.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_1_1,
                                                  args=(model_name, dataset_name, ''),
                                                  name="exp_setting_1_1")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 5 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                        f'data/outputs/{dataset_name}/{model_name}/step_9_shuffled_context_normed.csv'):
                    print(f'Warning: Norm for experimental setting 5 {dataset_name}/{model_name} Does not yet Exist')
            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_8_long_context.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_1_2_1,
                                                  args=(model_name, dataset_name, ''),
                                                  name="exp_setting_1_2_1")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 1_2_1 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                        f'data/outputs/{dataset_name}/{model_name}/step_8_long_context_normed.csv'):
                    print(
                        f'Warning: Norm for experimental setting 1_2_1 {dataset_name}/{model_name} Does not yet Exist')
            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_10_long_context_med.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_1_2_2,
                                                  args=(model_name, dataset_name, ''),
                                                  name="exp_setting_1_2_2")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 1_2_2 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                        f'data/outputs/{dataset_name}/{model_name}/step_10_long_context_med_normed.csv'):
                    print(
                        f'Warning: Norm for experimental setting 1_2_2 {dataset_name}/{model_name} Does not yet Exist')
            # -- Experimental Setting 2
            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_6_self_consistency_permuted_new.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_2_1,
                                              args=(model_name, dataset_name, ''),
                                              name="exp_setting_2_1")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 2_1 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                    f'data/outputs/{dataset_name}/{model_name}/step_6_self_consistency_permuted_new_normed.csv'):
                    print(f'Warning: Norm for experimental setting 2_1 {dataset_name}/{model_name} Does not yet Exist')

            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_7_self_consistency_permuted_no_context.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_2_2(),
                                                  args=(model_name, dataset_name, ''),
                                                  name="exp_setting_2_2")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 2_2 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                    f'data/outputs/{dataset_name}/{model_name}/step_7_self_consistency_permuted_no_context_normed.csv'):
                    print(f'Warning: Norm for experimental setting 2_2 {dataset_name}/{model_name} Does not yet Exist')

            # -- Experimental Setting 3
            if not os.path.exists(f'data/outputs/{dataset_name}/{model_name}/step_5_no_correct_answer.csv'):
                process = multiprocessing.Process(target=call_experimental_setting_3_all,
                                                  args=(model_name, dataset_name, ''),
                                                  name="exp_setting_3_all")
                process.start()
                process.join()
            else:
                print(f'Skipping experimental setting 2/5 for {dataset_name}/{model_name}: Already Exists')
                if not os.path.exists(
                    f'data/outputs/{dataset_name}/{model_name}/step_5_no_correct_answer_normed.csv'):
                    print(f'Warning: Norm for experimental setting 2/5 {dataset_name}/{model_name} Does not yet Exist')
