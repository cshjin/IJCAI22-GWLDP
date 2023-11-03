################################################################################
# Copyright (c) Samsung Research America, inc. All Rights Reserved.
#
# GNN model.
# Author: Hongwei Jin <hjin25@uic.edu>, 2021-06.
################################################################################

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch_geometric.nn import (GCNConv, GINConv, GraphConv, TopKPooling,
                                global_add_pool)
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GCN(Module):
    """ Graph classification using two-layer GCN and average pooling.

    Args:
        hidden (int): size of hidden layer
        n_features (int): size of feature dimension
        n_classes (int): number of classes
        act (str, optional): activation function. Defaults to 'relu'.
        pool (str, optional): pooling method, option of `avg`, `max`.
            Defaults to 'avg'.
        dropout (float, optional): dropout rate in training. Defaults to 0..
    """

    def __init__(self,
                 n_features,
                 hidden,
                 n_classes,
                 bias=True,
                 pool='avg',
                 dropout=0.):
        super(GCN, self).__init__()

        self.n_features = n_features
        self.hidden = hidden
        self.n_classes = n_classes
        self.pool = pool

        # GCN layer
        self.conv1 = GCNConv(self.n_features, self.hidden, normalize=True)
        self.conv2 = GCNConv(self.hidden, self.hidden, normalize=True)

        # linear output
        self.lin = Linear(self.hidden, self.n_classes)

        # dropout
        self.dropout = dropout

    def forward(self, data):
        """ Forward computation of model, computes the logits for each graph.

        Args:
            data (ptg.Data): graph data.

        Returns:
            torch.Tensor: logits of predictions for each label.
                Dimension of [B, K].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)

        if self.pool == 'avg':
            x = gap(x, batch)
        elif self.pool == 'max':
            x = gmp(x, batch)
        logits = self.lin(x)
        self.emb = logits
        return logits


class GIN(Module):
    def __init__(self, in_channels, dim, out_channels, dropout=0.5):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        return x

    def predict(self, data):
        """ Predict the classes of the input data

        Args:
            data (ptg.Data): graph data

        Returns:
            list: labels of the predicted data, [B, ]
        """
        return self.forward(data).argmax(1).detach()

    # TODO: verify the graph embedding in batch format
    def get_embed(self, data):
        """ Get the node embeddings of graph.

        Args:
            data (ptg.Data): graph data

        Returns:
            torch.Tensor: node embedding of graphs
        """
        self.eval()
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        return x

    def get_t(self, input, output):
        return output.data


def train(model, optimizer, loader):
    """ Train GC_Net.

    Args:
        model (nn.Module): GNN model.
        optimizer (torch.optim.Optimizer): model optimizer.
        loader (ptg.data.dataloader.DataLoader): dataloader of graphs.

    Returns:
        float: Loss value after training.
    """
    model.train()
    _device = next(model.parameters()).device

    loss_all = 0
    loader = loader

    for idx, data in enumerate(loader):
        data = data.to(_device)
        optimizer.zero_grad()
        output = model(data)
        logits = F.log_softmax(output, dim=1)
        # REVIEW: new in GIN
        # output = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        # loss = F.nll_loss(output, data.y)

        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()

    return loss_all / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    """ Evaluate model with dataloader

    Args:
        model (nn.Module): GNN model instance
        loader (ptg.data.dataloader.DataLoader): data loader

    Returns:
        (tuple): (accuracy, loss).
    """
    model.eval()
    _device = next(model.parameters()).device
    correct = 0
    loss_all = 0
    for data in loader:
        data = data.to(_device)
        # REVIEW: new in GIN
        # output = model(data.x, data.edge_index, data.batch)
        output = model(data)
        logits = F.log_softmax(output, dim=1)
        # loss
        loss = F.cross_entropy(logits, data.y)
        loss_all += data.num_graphs * loss.item()
        # accuracy
        pred = logits.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss_all / len(loader.dataset)
