################################################################################
# Copyright (c) Samsung Research America, inc. All Rights Reserved.
#
# GNN model.
# Author: Hongwei Jin <hjin25@uic.edu>, 2021-06.
################################################################################

import networkx as nx
import numpy as np
from ot.gromov import gromov_wasserstein
from scipy.sparse.csgraph import shortest_path


def node_dp(G, p=None):
    """ Node differential privacy with differ by one node.

    Args:
        G (nx.Graph): Graph representation.
        p (np.array, optional): Node distributions. Defaults to None.

    Returns:
        list: GW distances with all possible node DP changes.
    """
    res = []
    A_sp = nx.adjacency_matrix(G)
    C0 = shortest_path(A_sp, directed=False)
    n = A_sp.shape[0]

    if p is None:
        # uniform node distribution
        p = np.ones(n) / n
    for nd in G.nodes():
        G_new = G.copy()
        G_new.remove_node(nd)
        A_new = nx.adjacency_matrix(G_new)
        C1 = shortest_path(A_new, directed=False)
        m = A_new.shape[0]
        C1 = np.nan_to_num(C1, posinf=m)
        # C1 = C1 / np.linalg.norm(C1)
        q = np.ones(m) / m
        T, gw_log = gromov_wasserstein(C0, C1, p, q, loss_fun="square_loss", log=True)
        res.append(gw_log['gw_dist'])
    return res


def edge_dp(G, p=None):
    """ Edge differential privacy with differ by one edge.

    Args:
        G (nx.Graph): Graph representation.
        p (np.array, optional): Node distributions. Defaults to None.

    Returns:
        list: GW distances with all possible node DP changes.
    """
    res = []
    A_sp = nx.adjacency_matrix(G)
    C0 = shortest_path(A_sp, directed=False)
    n = A_sp.shape[0]

    if p is None:
        # uniform node distribution
        p = np.ones(n) / n
    for i in range(n):
        for j in range(i + 1, n):
            G_new = G.copy()
            if G.has_edge(i, j):
                G_new.remove_edge(i, j)
            else:
                G_new.add_edge(i, j)
            # largest_cc = max(nx.connected_components(G_new), key=len)
            # G_new = nx.subgraph(G_new, largest_cc)
            A_new = nx.adjacency_matrix(G_new)
            C1 = shortest_path(A_new, directed=False)
            m = A_new.shape[0]
            C1 = np.nan_to_num(C1, posinf=m)
            q = np.array(m) / m
            T, gw_log = gromov_wasserstein(C0, C1, p, q, loss_fun="square_loss", log=True)
            res.append(gw_log['gw_dist'])
    return res


def ldp_emb(embeds, eps=None):
    """ Local differential privacy on graph embedding.

    Args:
        embeds (list): List of graph embeddings.
        eps (float, optional): Epsilon noise added to embedding. Defaults to None.

    Returns:
        List: eps-LDP embeddings.
    """
    embeds_ldp = []
    for emb in embeds:
        if eps is None:
            eps = 1. / emb.shape[0]
        ldp = np.random.normal(0, eps, emb.shape)
        emb += ldp
        embeds_ldp.append(emb)
    return embeds_ldp


def ldp_D(D, eps=None):
    """ Local differential privacy on cost matrix.

    Args:
        embeds (np.ndarray): Cost matrix.
        eps (float, optional): Epsilon noise added to embedding. Defaults to None.

    Returns:
        np.ndarray: eps-LDP cost matrix.

    See Also:
        `gwldp.ldp.ldp_emb`
    """
    if eps is None:
        eps = 1. / D.shape[0]
    D += np.random.normal(0, eps, D.shape)
    D[D < 0] = 0
    D = abs(D)
    D = (D.T + D) / 2
    np.fill_diagonal(D, 0)
    return D
