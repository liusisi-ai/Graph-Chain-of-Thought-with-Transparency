import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from gmini import *
from utils import *
from model import *
from dataloader import *
from gcn import *



def main_pretrain(data, n_in, n_h, num_layers_num, dropout, negative_sample_num, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index_np = data.edge_index.cpu().numpy()
    negative_samples = prompt_pretrain_sample(edge_index_np, n=negative_sample_num)

    model = PrePrompt(n_in=n_in, n_h=n_h, num_layers_num=num_layers_num, dropout=dropout, sample=negative_samples).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    best_loss, best_model_state, best_embed = float('inf'), None, None

    for epoch in tqdm(range(1, epochs + 1), desc="Pretrain"):
        model.train()
        optimizer.zero_grad()
        loss = model(data.x, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch > epochs - 100 and loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            best_embed = model.embed(data.x, data.edge_index)

    if best_model_state:
        torch.save(best_model_state, CKPT_PATH)
        torch.save(best_embed, EMBED0_PATH)
    return model, best_embed


def main_downstream(gcn, label, gcn_model, data, n_in, n_h, hidden_dim, dropout, num_classes, downstream_epochs,
                    downstream_lr, thoughts, update_thought_every, task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = FusionMLP(hidden_dim=hidden_dim, n_h=n_h, n_in=n_in, dropout=dropout, num_classes=num_classes,
                      num_nodes=data.num_nodes,data=data).to(device)
    optimizer = optim.Adam(model.parameters(), lr=downstream_lr)
    criterion_lp = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, downstream_epochs + 1), desc="Downstream"):
        model.train()
        optimizer.zero_grad()

        update_thought = (epoch - 1) % update_thought_every == 0 and (epoch // update_thought_every) < thoughts
        new_x, logits = model(gcn, data.edge_index, data.x, update_thought, epoch)

        if task == "nc":
            loss = nn.CrossEntropyLoss()(logits[data.train_mask], data.y[data.train_mask])
        elif task == "lp":
            pos_edge = data.edge_index
            neg_edge = negative_sampling(edge_index=pos_edge, num_nodes=data.num_nodes,
                                         num_neg_samples=pos_edge.size(1)).to(device)
            pos_pred = model.predict_links(new_x, pos_edge)
            neg_pred = model.predict_links(new_x, neg_edge)
            loss = criterion_lp(torch.cat([pos_pred, neg_pred]),
                                torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).to(device))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == downstream_epochs:
            model.eval()
            with torch.no_grad():
                cur_x, cur_logits = model(gcn, data.edge_index, data.x, False, epoch)

                if task == "nc":
                    val_pred = cur_logits[data.val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                    test_pred = cur_logits[data.test_mask].argmax(dim=1)
                    test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
                    print(f" Epoch {epoch:03d} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
                elif task == "lp":
                    val_p_score = model.predict_links(cur_x, data.val_pos_edge_index).sigmoid()
                    val_n_score = model.predict_links(cur_x, data.val_neg_edge_index).sigmoid()
                    val_y_pred = torch.cat([val_p_score, val_n_score]).cpu().numpy()
                    val_y_true = np.hstack([np.ones(val_p_score.size(0)), np.zeros(val_n_score.size(0))])
                    val_auc = roc_auc_score(val_y_true, val_y_pred)
                    val_ap = average_precision_score(val_y_true, val_y_pred)
                    test_p_score = model.predict_links(cur_x, data.test_pos_edge_index).sigmoid()
                    test_n_score = model.predict_links(cur_x, data.test_neg_edge_index).sigmoid()
                    test_y_pred = torch.cat([test_p_score, test_n_score]).cpu().numpy()
                    test_y_true = np.hstack([np.ones(test_p_score.size(0)), np.zeros(test_n_score.size(0))])
                    test_auc = roc_auc_score(test_y_true, test_y_pred)
                    test_ap = average_precision_score(test_y_true, test_y_pred)
                    print(
                        f" Epoch {epoch:03d} | Val AUC: {val_auc:.4f} AP: {val_ap:.4f} | Test AUC: {test_auc:.4f} AP: {test_ap:.4f}")

        return cur_x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TASK = "lp"

    if TASK == "lp":
        data = load_lp_data_with_test_split("cora")
        NUM_CLASSES = 1
    else:
        data = load_gnn_dataset("cora", task="nc")
        NUM_CLASSES = len(torch.unique(data.y))


    n_in, n_h, num_layers_num, dropout = 128, 128, 2, 0.
    pretrain_epochs, pretrain_lr = 1000, 0.01
    downstream_epochs, downstream_lr = 400, 0.01
    thoughts, update_thought_every = 2, 100

    data.x = torch.FloatTensor(pca_compression(data.x, k=n_in))
    os.makedirs(f'{DATASET_NAME}_checkpoints', exist_ok=True)
    if not os.path.exists(CKPT_PATH):
        gcn_model, _ = main_pretrain(data, n_in, n_h, num_layers_num, dropout, 2, pretrain_epochs, pretrain_lr)
    else:
        gcn_model = PrePrompt(n_in=n_in, n_h=n_h, num_layers_num=num_layers_num, dropout=dropout).to(device)
        gcn_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    main_downstream(
        gcn_model.gcn, data.y, gcn_model, data, n_in, n_h, 256, dropout,
        NUM_CLASSES, downstream_epochs, downstream_lr, thoughts, update_thought_every, task=TASK
    )
    print("✅ Process Completed.")


if __name__ == '__main__':
    main()