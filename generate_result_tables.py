import os.path

import pandas as pd

import global_values

if __name__ == '__main__':
    # Experiment 1:
    print("==== Experimental Setting 1 =====")
    data = [
        ["Dataset", "Setting"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]

    for dataset in global_values.datasets:
        output_row_correct = [dataset, "Correct"]
        output_row_incorrect = [dataset, "Incorrect"]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_6_self_consistency_permuted_new_normed.csv'):
                output_row_correct.append('')
                output_row_incorrect.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_6_self_consistency_permuted_new_normed.csv', true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['permuted_correct_answer'] == 'A']
            df_correct_ans_correct = df_correct_ans[df_correct_ans['permuted_answer_correct'] == True]
            output_row_correct.append(f'{df_correct_ans_correct.shape[0]}/{df_correct_ans.shape[0]} ({df_correct_ans_correct.shape[0]/df_correct_ans.shape[0]:.2%})')
            df_incorrect_ans = df[df['permuted_correct_answer'] == 'B']
            df_incorrect_ans_correct = df_incorrect_ans[df_incorrect_ans['permuted_answer_correct'] == True]
            output_row_incorrect.append(f'{df_incorrect_ans_correct.shape[0]}/{df_incorrect_ans.shape[0]} ({df_incorrect_ans_correct.shape[0]/df_incorrect_ans.shape[0]:.2%})')
        data.append(output_row_correct)
        data.append(output_row_incorrect)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<20} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}  {row[6]:<30}")


    # Experiment 2:
    print("==== Experimental Setting 2 =====")
    data = [
        ["Dataset"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]

    for dataset in global_values.datasets:
        output_row = [dataset]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_5_no_correct_answer_normed.csv'):
                output_row.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_5_no_correct_answer_normed.csv',
                             true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['no_valid_option_explicit_answer_correct'] == True]
            output_row.append(
                f'{df_correct_ans.shape[0]}/{df.shape[0]} ({df_correct_ans.shape[0] / df.shape[0]:.2%})')
        data.append(output_row)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<30} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}")

    # Experiment 3
    print("==== Experimental Setting 3 =====")
    data = [
        ["Dataset"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]

    for dataset in global_values.datasets:
        output_row = [dataset]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_8_long_context_normed.csv'):
                output_row.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_8_long_context_normed.csv',
                             true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['expanded_context_answer_correct'] == True]
            output_row.append(
                f'{df_correct_ans.shape[0]}/{df.shape[0]} ({df_correct_ans.shape[0] / df.shape[0]:.2%})')
        data.append(output_row)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<30} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}")

    # Experiment 4
    print("==== Experimental Setting 4 =====")
    data = [
        ["Dataset", "Setting"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]
    for dataset in global_values.datasets:
        output_row_correct = [dataset, "Correct"]
        output_row_incorrect = [dataset, "Incorrect"]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_7_self_consistency_permuted_no_context_normed.csv'):
                output_row_correct.append('')
                output_row_incorrect.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_7_self_consistency_permuted_no_context_normed.csv',
                             true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['permuted_correct_answer'] == 'A']
            df_correct_ans_correct = df_correct_ans[df_correct_ans['permuted_answer_correct'] == True]
            output_row_correct.append(
                f'{df_correct_ans_correct.shape[0]}/{df_correct_ans.shape[0]} ({df_correct_ans_correct.shape[0] / df_correct_ans.shape[0]:.2%})')
            df_incorrect_ans = df[df['permuted_correct_answer'] == 'B']
            df_incorrect_ans_correct = df_incorrect_ans[df_incorrect_ans['permuted_answer_correct'] == True]
            output_row_incorrect.append(
                f'{df_incorrect_ans_correct.shape[0]}/{df_incorrect_ans.shape[0]} ({df_incorrect_ans_correct.shape[0] / df_incorrect_ans.shape[0]:.2%})')
        data.append(output_row_correct)
        data.append(output_row_incorrect)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<20} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}  {row[6]:<30}")

    # Experiment 5:
    print("==== Experimental Setting 5 =====")
    data = [
        ["Dataset"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]

    for dataset in global_values.datasets:
        output_row = [dataset]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_5_no_correct_answer_normed.csv'):
                output_row.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_5_no_correct_answer_normed.csv',
                             true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['no_valid_option_nonexplicit_answer_correct'] == True]
            output_row.append(
                f'{df_correct_ans.shape[0]}/{df.shape[0]} ({df_correct_ans.shape[0] / df.shape[0]:.2%})')
        data.append(output_row)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<30} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}")

    # Experiment 6
    print("==== Experimental Setting 6 =====")
    data = [
        ["Dataset"] + list(map(lambda x: x.split('/')[1], global_values.models))
    ]

    for dataset in global_values.datasets:
        output_row = [dataset]
        for model in global_values.models:
            if not os.path.exists(f'data/outputs/{dataset}/{model}/step_9_shuffled_context_normed.csv'):
                output_row.append('')
                continue
            df = pd.read_csv(f'data/outputs/{dataset}/{model}/step_9_shuffled_context_normed.csv',
                             true_values=["TRUE"], false_values=["FALSE"])
            df_correct_ans = df[df['shuffled_context_answer_correct'] == True]
            output_row.append(
                f'{df_correct_ans.shape[0]}/{df.shape[0]} ({df_correct_ans.shape[0] / df.shape[0]:.2%})')
        data.append(output_row)

    for row in data:
        print(f"{row[0]:<30} {row[1]:<30} {row[2]:<30} {row[3]:<30} {row[4]:<30} {row[5]:<30}")