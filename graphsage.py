"""GraphSAGE model + contrastive self-supervised pre-training.

Replaces the original GCN stack for the first-stage pre-training.  The
pre-trained model is saved as ``first_model`` and later loaded as a graph
encoder in the CoT training stage.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from gcn import compareloss, prompt_pretrain_sample


class SAGELayers(nn.Module):
    """Stacked GraphSAGE convolution layers (mirrors the old GcnLayers API)."""

    def __init__(self, n_in: int, n_h: int, num_layers_num: int, dropout: float):
        super().__init__()
        self.num_layers_num = num_layers_num
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList()
        for i in range(num_layers_num):
            in_dim = n_in if i == 0 else n_h
            self.convs.append(SAGEConv(in_dim, n_h, aggr="mean"))

        self.input_proj = (
            nn.Linear(n_in, n_h) if n_in != n_h else nn.Identity()
        )

    def forward(self, x, edge_index):
        h = None
        for i, conv in enumerate(self.convs):
            if i == 0:
                h = conv(x, edge_index)
            else:
                h = conv(h, edge_index) + h  # residual
            h = self.act(h)
            h = self.dropout(h)
        return h


class PrePromptSAGE(nn.Module):
    """GraphSAGE encoder + projector for contrastive pre-training.

    The projector maps graph embeddings to an LLM-aligned hidden space
    (``projector_out_dim``).  During the first stage the projector is only
    warmed up; in the second stage it is fine-tuned for LLM-dim alignment.
    """

    def __init__(
        self,
        n_in: int,
        n_h: int,
        num_layers_num: int,
        dropout: float,
        projector_out_dim: int = 4096,
        sample=None,
    ):
        super().__init__()
        self.sage = SAGELayers(n_in, n_h, num_layers_num, dropout)
        self.projector = nn.Sequential(
            nn.Linear(n_h, projector_out_dim),
            nn.GELU(),
            nn.Linear(projector_out_dim, projector_out_dim),
        )
        if sample is not None:
            self.negative_sample = torch.tensor(
                sample, dtype=torch.int64
            ).cuda() if torch.cuda.is_available() else torch.tensor(
                sample, dtype=torch.int64
            )
        else:
            self.negative_sample = None
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        g = self.sage(x, edge_index)
        loss = compareloss(g, self.negative_sample, temperature=1)
        return loss

    def embed(self, x, edge_index):
        g = self.sage(x, edge_index)
        return g.detach()

    def embed_with_projection(self, x, edge_index):
        h = self.sage(x, edge_index)
        p = self.projector(h)
        return h, p


def pretrain_sage(
    data,
    n_in: int,
    n_h: int,
    num_layers_num: int,
    dropout: float,
    negative_sample_num: int,
    epochs: int,
    lr: float,
    projector_out_dim: int,
    save_path: str,
):
    """Run contrastive self-supervised training on GraphSAGE, save first_model."""
    from tqdm import tqdm
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index_np = data.edge_index.cpu().numpy()
    negative_samples = prompt_pretrain_sample(edge_index_np, n=negative_sample_num)

    model = PrePromptSAGE(
        n_in=n_in,
        n_h=n_h,
        num_layers_num=num_layers_num,
        dropout=dropout,
        projector_out_dim=projector_out_dim,
        sample=negative_samples,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    best_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(1, epochs + 1), desc="SAGE Pretrain"):
        model.train()
        optimizer.zero_grad()
        loss = model(data.x, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch > epochs - 100 and loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_state is not None:
        torch.save(best_state, save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"✅ first_model saved to {save_path} (best loss: {best_loss:.4f})")
    return model
