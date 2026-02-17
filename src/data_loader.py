# src/data_loader.py

import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def basic_cleaning(self, df):
        # Drop duplicates
        df = df.drop_duplicates()
        # Handle missing values (example: fill with median or mode)
        df = df.fillna(df.median(numeric_only=True))
        return df
