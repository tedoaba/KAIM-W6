import pandas as pd

def load_data(filepath):
    """Load dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df
