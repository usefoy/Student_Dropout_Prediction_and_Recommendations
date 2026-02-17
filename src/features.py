# src/features.py

from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.scaler = None

    def encode_categoricals(self, df, categorical_cols):
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def scale_features(self, df, numeric_cols):
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
