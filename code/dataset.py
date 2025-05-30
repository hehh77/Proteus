import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

class ProteomicsDataset(Dataset):
    def __init__(self, csv_path, protein_embed_npy=None):
        super().__init__()
        
        df = pd.read_csv(csv_path)
        
        cols_to_skip = ["cellLabel", "cellSize", "cellType", 
                        "cellTypeIdx", "X_cent", "Y_cent"]                     
        
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c not in cols_to_skip]
        
        mean_x = df["X_cent"].mean()
        std_x = df["X_cent"].std() + 1e-9 
        mean_y = df["Y_cent"].mean()
        std_y = df["Y_cent"].std() + 1e-9

        df["x_norm"] = (df["X_cent"] - mean_x) / std_x
        df["y_norm"] = (df["Y_cent"] - mean_y) / std_y   

        self.labels = df["cellTypeIdx"].values
        
        if protein_embed_npy is not None and os.path.exists(protein_embed_npy):
            protein_embeds_dict = np.load(protein_embed_npy, allow_pickle=True).item()
        else:
            print(f"Warning: Embedding file not found at {protein_embed_npy}. Using random embeddings.")
            embed_dim = 320 
            protein_embeds_dict = {col: np.random.randn(embed_dim) for col in feature_cols}
        
        # Ensure all feature columns have an embedding
        embed_dim = len(list(protein_embeds_dict.values())[0])
        mean_val = np.mean(list(protein_embeds_dict.values()))
        std_val = np.std(list(protein_embeds_dict.values()))
        
        valid_cols = []
        final_embeddings = []
        for c in feature_cols:
            if c in protein_embeds_dict:
                final_embeddings.append(protein_embeds_dict[c])
                valid_cols.append(c)
            else:
                emb_temp = np.random.randn(embed_dim) * std_val + mean_val
                final_embeddings.append(emb_temp)
                valid_cols.append(c)

        self.feature_cols = valid_cols
        self.marker_embeds = np.array(final_embeddings)

        self.P = len(valid_cols)
        self.embed_dim = self.marker_embeds.shape[1]

        self.X_expr = df[self.feature_cols].values
        self.XY = df[["x_norm", "y_norm"]].values

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x_expr = self.X_expr[idx]
        xy = self.XY[idx]
        y = self.labels[idx]
        
        x_expr = torch.tensor(x_expr, dtype=torch.float32)
        xy = torch.tensor(xy, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return x_expr, xy, y
    
    def get_marker_embeds(self):
        """Returns initial marker embeddings as a tensor."""
        return torch.tensor(self.marker_embeds, dtype=torch.float32)
    
    def get_feature_names(self):
        """Returns the names of the marker features used."""
        return self.feature_cols