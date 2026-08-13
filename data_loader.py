from datasets import load_dataset

def load_code_data(language="python", split="train", limit=None):
    """Loads dataset and extracts the 'code' and 'label' columns."""
    dataset = load_dataset("HungPhamBKCS/magecode-dataset", name=language, split=split)
    
    # Using 'code' instead of 'text'
    codes = dataset['code'][:limit] if limit else dataset['code']
    labels = dataset['label'][:limit] if limit else dataset['label']
    
    return codes, labels
