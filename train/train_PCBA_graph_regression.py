"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import numpy as np
from ogb.graphproppred import Evaluator
from tqdm import tqdm

"""
    For GCNs
"""


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_AP = 0
    list_scores = []
    list_labels = []
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
        is_labeled = batch_targets == batch_targets
        loss = model.loss(batch_scores[is_labeled], batch_targets.float()[is_labeled])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        list_scores.append(batch_scores.detach().cpu())
        list_labels.append(batch_targets.detach().cpu())

    epoch_loss /= (iter + 1)
    evaluator = Evaluator(name='ogbg-molpcba')
    print(torch.cat(list_scores).size())
    print(torch.cat(list_labels).size())
    print(torch.cat(list_scores))
    print(torch.cat(list_labels))
    epoch_train_AP = evaluator.eval({'y_pred': torch.cat(list_scores), 'y_true': torch.cat(list_labels)})['ap']

    return epoch_loss, epoch_train_AP, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_AP = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            is_labeled = batch_targets == batch_targets
            loss = model.loss(batch_scores[is_labeled], batch_targets.float()[is_labeled])
            epoch_test_loss += loss.detach().item()
            list_scores.append(batch_scores.detach().cpu())
            list_labels.append(batch_targets.detach().cpu())

        epoch_test_loss /= (iter + 1)
        evaluator = Evaluator(name='ogbg-molpcba')
        epoch_test_AP = evaluator.eval({'y_pred': torch.cat(list_scores), 'y_true': torch.cat(list_labels)})['ap']

    return epoch_test_loss, epoch_test_AP