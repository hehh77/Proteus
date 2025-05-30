import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import logging
import os
from torch.utils.data import DataLoader, random_split

from utils import create_experiment_folders, set_seed, setup_logger,evaluate_model
from dataset import ProteomicsDataset
from model import Proteus



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    folders = create_experiment_folders()
    logger = setup_logger(folders['logs'])
    
    logger.info(f"Experiment folders created at: {folders['base']}")
    
    config = {
        'batch_size': 128,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'lr': 4e-4,
        'weight_decay': 1e-3,
        'epochs': 200,
        'patience': 20,
        'label_smoothing': 0.1,
        'coord_embed_dim': 128
    }
    logger.info(f"Configuration: {config}")
    

    num_folds = 5
    for fold in range(1, num_folds + 1):
        logger.info(f"\n===== Processing Fold {fold}/{num_folds} =====")
        

        data_base_path = '../data/CODEX' 
        train_csv = os.path.join(data_base_path, f'fold/fold_{fold}_training.csv')
        test_csv = os.path.join(data_base_path, f'fold/fold_{fold}_test.csv')
        marker_embed_npy = os.path.join(data_base_path, 'marker_embedding/CODEX_marker_embeddings.npy')

        
        train_full_dataset = ProteomicsDataset(train_csv, marker_embed_npy) 
        test_dataset = ProteomicsDataset(test_csv, marker_embed_npy)
        
        # Split training data into train and validation sets
        train_size = int(0.875 * len(train_full_dataset)) 
        val_size = len(train_full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) 
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        

        #Model Initialization
        initial_marker_embeds = train_full_dataset.get_marker_embeds()
        P = initial_marker_embeds.shape[0]
        marker_embed_dim = initial_marker_embeds.shape[1]
        num_classes = len(pd.read_csv(train_csv)["cellTypeIdx"].unique()) 
        

        model = Proteus(
            P=P,
            marker_embed_dim=marker_embed_dim,
            initial_marker_embeds=initial_marker_embeds,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            dropout=config['dropout'],
            coord_embed_dim=config['coord_embed_dim']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=10,
            verbose=False
        )

        best_val_f1 = -1.0
        patience_counter = 0
        best_epoch = -1
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0.0
            num_batches = len(train_loader)
            
            for i, (x_expr, xy, labels) in enumerate(train_loader):
                x_expr, xy, labels = x_expr.to(device), xy.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x_expr, xy)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / num_batches
            
            # Validation
            val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, device)
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Fold {fold}, Epoch {epoch+1}/{config['epochs']} - LR: {current_lr:.6f}, "
                       f"Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            scheduler.step(val_f1)


            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_metrics': (val_acc, val_prec, val_rec, val_f1)
                }, f"{folders['models']}/best_model_fold{fold}.pt")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    logger.info(f"Early stopping triggered")
                    break
        
        best_model_dict = torch.load(f"{folders['models']}/best_model_fold{fold}.pt")
        model.load_state_dict(best_model_dict['model_state_dict'])
        

        test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, device)
        
        
        logger.info(f"\nFold {fold} Final Results:")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Precision: {test_prec:.4f}")
        logger.info(f"Test Recall: {test_rec:.4f}")
        logger.info(f"Test Macro F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()