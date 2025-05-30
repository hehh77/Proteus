import torch
import torch.nn as nn

class EMInteractorBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, ff_ratio=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=num_heads,
                                                batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        ff_hidden = ff_ratio * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v)
        q = q + self.dropout(attn_out)
        q = self.attn_ln(q)

        ff_out = self.ff(q)
        q = q + self.dropout(ff_out)
        q = self.ff_ln(q)

        return q, attn_weights


class Proteus(nn.Module):
    def __init__(self, 
                 P,                     # Number of protein/marker features
                 marker_embed_dim,      # Dimension of initial marker embeddings
                 initial_marker_embeds, # Initial marker embeddings
                 d_model=128,           # Internal dimension of the model
                 num_heads=8,           # Number of attention heads
                 num_layers=6,          # Number of EMInteractorBlocks
                 num_classes=18,        # Number of output cell types
                 dropout=0.1,
                 coord_embed_dim=128):  # Dimension for spatial coordinate embedding
        super().__init__()

        self.spatial_embedding_projection = nn.Sequential(
            nn.Linear(2, coord_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(coord_embed_dim, coord_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.marker_embeddings = nn.Parameter(
            initial_marker_embeds.clone(),
            requires_grad=True
        )
        
        input_dim_to_q = P + coord_embed_dim

        self.query_projection = nn.Sequential(
            nn.Linear(input_dim_to_q, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        hidden_dim_kv = 256 
        self.key_projection = nn.Sequential(
            nn.Linear(marker_embed_dim, hidden_dim_kv),
            nn.LayerNorm(hidden_dim_kv),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim_kv, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(marker_embed_dim, hidden_dim_kv),
            nn.LayerNorm(hidden_dim_kv),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim_kv, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.em_interactor_blocks = nn.ModuleList(
            [EMInteractorBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) 
             for _ in range(num_layers)]
        )

        self.post_ln = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x_expr, xy):
        bs = x_expr.size(0)
        
        e_xy = self.spatial_embedding_projection(xy)
        x_cat = torch.cat([x_expr, e_xy], dim=-1)

        q = self.query_projection(x_cat)
        q = q.unsqueeze(1) 

        current_marker_embeds = self.marker_embeddings.unsqueeze(0).expand(bs, -1, -1)
        k = self.key_projection(current_marker_embeds) 
        v = self.value_projection(current_marker_embeds) 

        attn_weights_last = None
        for block in self.em_interactor_blocks:
            q, attn_weights_last = block(q, k, v)
        
        q = self.post_ln(q)

        cell_embedding = q.squeeze(1) 

        logits = self.classifier(cell_embedding)
        
        return logits, attn_weights_last