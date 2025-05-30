import torch
import numpy as np
import logging
import os
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def create_experiment_folders(base_name="CODEX"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{base_name}_{timestamp}"
    base_dir = f"../Result/{experiment_name}"
    
    folders = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        
    return folders

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logger(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def evaluate_model(model, data_loader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_expr, xy, labels in data_loader:
            x_expr, xy, labels = x_expr.to(device), xy.to(device), labels.to(device)
            # Proteus model expects xy coordinates
            logits, _ = model(x_expr, xy) 
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0 
    ) 
    
    return accuracy, precision, recall, f1