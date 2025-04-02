import pandas as pd
import numpy as np
from scipy.stats import boxcox

def preprocess_input(form_data, pipeline):
    # Convert to DataFrame
    input_df = pd.DataFrame([form_data])
    
    # Define feature types
    continuous = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical = ['sex', 'fbs', 'exang', 'slope', 'ca']
    
    # Type conversion
    for col in continuous + categorical:
        input_df[col] = pd.to_numeric(input_df[col])
    
    # Box-Cox transform
    input_df['oldpeak'] = input_df['oldpeak'].astype(float) + 0.001
    for col in continuous:
        if col in pipeline['boxcox_lambdas'] and input_df[col].min() > 0:
            input_df[col] = boxcox(input_df[col], lmbda=pipeline['boxcox_lambdas'][col])
    
    # One-hot encoding
    cp_val = str(int(form_data.get('cp', 1)))
    restecg_val = str(int(form_data.get('restecg', 0)))
    thal_val = str(int(form_data.get('thal', 1)))
    
    input_df['cp_1'] = 1 if cp_val in ['2', '3'] else 0
    input_df['cp_2'] = 1 if cp_val == '3' else 0
    input_df['restecg_1'] = 1 if restecg_val == '1' else 0
    input_df['thal_1'] = 1 if thal_val in ['2', '3'] else 0
    
    # Standard scaling
    input_df[continuous] = pipeline['scaler'].transform(input_df[continuous])
    
    # Ensure correct feature order
    return input_df[pipeline['feature_order']]