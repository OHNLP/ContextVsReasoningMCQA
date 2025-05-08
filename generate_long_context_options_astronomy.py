import pandas as pd

if __name__ == "__main__":
    df = pd.read_parquet("hf://datasets/joey234/mmlu-astronomy/data/test-00000-of-00001-34ef4ba0b3d351c3.parquet")
    df = pd.DataFrame(df["choices"].explode().tolist())
    df = df[df[0].str.split().apply(len) < 5]
    df.to_csv('data/input/expanded_context_opt_medical.txt', index=False, header=False)