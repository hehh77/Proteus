import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

def prepare_folds(data_path, output_dir, n_folds=5, seed=42):

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    unique_cell_types = df['cellType'].unique()
    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(sorted(unique_cell_types))}
    
    df['cellTypeIdx'] = df['cellType'].map(cell_type_to_idx)
    
    cell_type_mapping = pd.DataFrame({
        'cellType': list(cell_type_to_idx.keys()),
        'cellTypeIdx': list(cell_type_to_idx.values())
    })
    mapping_path = os.path.join(output_dir, 'cell_type_mapping.csv')
    cell_type_mapping.to_csv(mapping_path, index=False)
    print(f"Cell type mapping saved to: {mapping_path}")
    
    np.random.seed(seed)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(df))
    fold_stats = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), 1):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        train_dist = train_data['cellType'].value_counts().to_dict()
        test_dist = test_data['cellType'].value_counts().to_dict()
        
        train_path = os.path.join(output_dir, f'fold_{fold}_training.csv')
        test_path = os.path.join(output_dir, f'fold_{fold}_test.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        



data_path ='../data/CODEX/CODEX_annotation.csv'
output_dir = '../data/CODEX/fold'
prepare_folds(data_path, output_dir, n_folds=5, seed=42)