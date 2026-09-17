import pandas as pd
from datasets import load_dataset

def load_code_data(language="python", split="train", limit=None):
    """Loads dataset and extracts the 'code' and 'label' columns, applying C++ test balancing."""
    dataset = load_dataset("HungPhamBKCS/magecode-dataset", name=language, split=split)
    
    df = dataset.to_pandas()
    
    # Paper Sec 4.1 requires balancing C++ test set to 2998/2998
    if language == "cpp" and split == "test":
        human_df = df[df['label'] == 0]
        machine_df = df[df['label'] == 1]
        
        n_machine = len(machine_df)
        human_df = human_df.sample(n=n_machine, random_state=42)
        
        df = pd.concat([human_df, machine_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    codes = df['code'].tolist()
    labels = df['label'].tolist()
    
    if limit:
        codes = codes[:limit]
        labels = labels[:limit]
        
    return codes, labels
