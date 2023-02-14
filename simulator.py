#!/usr/bin/env python3
import os, random, logging, pickle, itertools
from itertools import compress
from collections import Counter
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import subgraph, coalesce, to_networkx

from utils.gcn_utils import GraphData, to_adj_lists, to_adj_nodes_with_times

### Helper Functions 

def to_lists(edge_index, num_nodes):
    G_out = dict([(i, []) for i in range(num_nodes)])
    G_in = dict([(i, []) for i in range(num_nodes)])
    for u,v in edge_index.T:
        u,v = int(u), int(v)
        G_out[u] += [v]
        G_in[v] += [u]
    G_in = dict(zip(G_in.keys(), [torch.LongTensor(v) for v in G_in.values()]))
    G_out = dict(zip(G_out.keys(), [torch.LongTensor(v) for v in G_out.values()]))
    return G_in, G_out

def get_median(data, func):
    y = func(data)
    return y.median()

def threshhold(func):
    def inner(data, threshhold=None):
        y = func(data)
        if threshhold is not None:
            y = y>threshhold
        return y.float()
    return inner

### Pattern Checking Functions 

@ threshhold
def deg_in(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_in[node]) for node in range(len(g_in))]).reshape((len(g_in), 1))

@ threshhold
def deg_out(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_out[node]) for node in range(len(g_out))]).reshape((len(g_out), 1))

@ threshhold
def fan_in(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_in[node].unique()) for node in range(len(g_in))]).reshape((len(g_in), 1))

@ threshhold
def fan_out(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_out[node].unique()) for node in range(len(g_out))]).reshape((len(g_out), 1))

@ threshhold
def max_fan_in(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    fan_ins = torch.Tensor([len(g_in[node].unique()) for node in range(len(g_in))]).reshape((len(g_in), 1))
    return torch.Tensor([ max([fan_ins[u] for u in g_in[node]] + [0]) for node in range(len(g_in)) ]).reshape((len(g_in), 1))

@ threshhold
def max_deg_in(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    deg_ins = torch.Tensor([len(g_in[node]) for node in range(len(g_in))]).reshape((len(g_in), 1))
    return torch.Tensor([ max([deg_ins[u] for u in g_in[node]] + [0]) for node in range(len(g_in)) ]).reshape((len(g_in), 1))

@ threshhold
def ratio_in(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_in[node])/ (len(g_in[node].unique())+0.0001) for node in range(len(g_in))]).reshape((len(g_in), 1))

@ threshhold
def ratio_out(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_out[node])/ (len(g_out[node].unique())+0.0001) for node in range(len(g_out))]).reshape((len(g_out), 1))

@ threshhold
def deg_in_timespan(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    max_time = data.timestamps.max()
    edge_index = edge_index[:, data.timestamps < 0.5 * max_time]
    g_in, g_out = to_lists(edge_index, num_nodes)
    return torch.Tensor([len(g_in[node]) for node in range(len(g_in))]).reshape((len(g_in), 1))

@ threshhold
def timevar_in(data):
    adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
    labels = [0 for i in range(len(adj_list_in))]
    for i in range(len(adj_list_in)):
        if len(adj_list_in[i]) > 1:
            labels[i] = np.array(adj_list_in[i]).std(axis=0)[1]
    return torch.Tensor(labels).reshape((len(adj_list_in), 1))

@ threshhold
def timevar_out(data):
    adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
    labels = [0 for i in range(len(adj_list_out))]
    for i in range(len(adj_list_out)):
        if len(adj_list_out[i]) > 1:
            labels[i] = np.array(adj_list_out[i]).std(axis=0)[1]
    return torch.Tensor(labels).reshape((len(adj_list_out), 1))

@ threshhold
def e_time_mod7(data):
    if data.edge_attr is None:
        logging.info(f"No Timestamps!!")
        return torch.zeros((data.edge_index.shape[1], 1))
    timestamps = data.edge_attr[:,0].reshape((-1,1))
    timestamps2 = data.edge_attr[:,1].reshape((-1,1))
    return torch.sin((timestamps + timestamps2))

@ threshhold
def C2_check(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        if len(list(set(g_out[i].numpy()) & set(g_in[i].numpy())-{i})) > 0:
            labels[i] = 1
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def C2_count(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        labels[i] = len(list(set(g_out[i].numpy()) & set(g_in[i].numpy())-{i}))
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def C3_check(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        for j in g_out[i]:
            if int(j) == i: continue
            for k in g_in[i]:
                if int(k) == i: continue
                if j != k and int(k) in g_out[int(j)]:
                    labels[i] = 1
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def C3_count(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        for j in g_out[i]:
            if int(j) == i: continue
            for k in g_in[i]:
                if int(k) == i: continue
                if j != k and int(k) in g_out[int(j)]:
                    labels[i] += 1
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def C4_check(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        for j in g_out[i]:
            if int(j) == i: continue
            for k in g_in[i]:
                if int(k) == i: continue
                if j != k:
                    if len(list(set(g_out[int(j)].numpy()) & set(g_in[int(k)].numpy())-{i, int(j), int(k)})) > 0:
                        labels[i] = 1
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def C4_count(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        for j in g_out[i]:
            for k in g_in[i]:
                if j != k:
                    labels[i] += len(list(set(g_out[int(j)].numpy()) & set(g_in[int(k)].numpy())-{i, int(j), int(k)}))
    return torch.Tensor(labels).reshape((len(g_out), 1))

def Cn_check_recursive(g_in, g_out, end, stoplist, n):
    end_in, end_out = end
    if n == 1:
        if end_in in g_out[end_out].numpy():
            return True
    if n == 2:
        if len( list( (set(g_out[end_out].numpy()) & set(g_in[end_in].numpy())) - stoplist - {end_in,end_out} ) ) > 0:
            return True
    if n > 2:
        for j in set(g_in[end_in].numpy()) - stoplist:
            stoplist_new = stoplist | {j}
            for k in set(g_out[end_out].numpy()) - stoplist_new:
                stoplist_new = stoplist | {j,k}
                end_new = (j,k)
                n_new = n - 2
                if Cn_check_recursive(g_in, g_out, end_new, stoplist_new, n_new):
                    return True
    return False

def Cn_check(n=3):
    @ threshhold
    def Cn_check_tmp(data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)
        labels = [0 for i in range(len(g_out))]
        for i in range(len(g_out)):
            end = (i,i)
            stoplist = {i}
            if Cn_check_recursive(g_in, g_out, end, stoplist, n):
                labels[i] = 1
        return torch.Tensor(labels).reshape((len(g_out), 1))
    return Cn_check_tmp

def Kn_check(n=3):
    @ threshhold
    def Kn_check_tmp(data):
        g = g = nx.from_edgelist(data.edge_index.T.numpy())
        N = data.num_nodes
        labels = [0 for i in range(N)]
        for clique in nx.find_cliques(g):
            if len(clique) >= n:
                for i in clique:
                    labels[i] = 1
        return torch.Tensor(labels).reshape((N, 1))
    return Kn_check_tmp

def Kn_count(n=3):
    @ threshhold
    def Kn_count_tmp(data):
        g = g = nx.from_edgelist(data.edge_index.T.numpy())
        N = data.num_nodes
        labels = [0 for i in range(N)]
        for clique in nx.find_cliques(g):
            if len(clique) == n:
                for i in clique:
                    labels[i] += 1
        return torch.Tensor(labels).reshape((N, 1))
    return Kn_count_tmp

def SGn_check(n=3):
    @ threshhold
    def SG_check_tmp(data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)
        labels = [0 for i in range(len(g_out))]
        for i in range(len(g_out)):
            if g_in[i].shape[0] > 0:
                two_hop_nbhd = torch.cat([g_in[int(j)] for j in g_in[i]]).unique()
                scatters = [len(list(set(g_out[int(k)].numpy()) & set(g_in[int(i)].numpy())-{i,k})) for k in two_hop_nbhd if k != i]
                if len(scatters) > 0:
                    if max(scatters) >= n:
                        labels[i] = 1
        return torch.Tensor(labels).reshape((len(g_out), 1))
    return SG_check_tmp


@ threshhold
def SG2_check(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        if g_in[i].shape[0] > 0:
            if (torch.cat([g_in[int(j)][torch.where(g_in[int(j)]!=i)].unique() for j in g_in[i].unique()]).unique(return_counts=True)[1] > 1).sum() > 0:
                labels[i] = 1
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def BP2_check(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        if g_in[i].shape[0] > 0:
            if (torch.cat([g_out[int(j)][torch.where(g_out[int(j)]!=i)].unique() for j in g_in[i].unique()]).unique(return_counts=True)[1] > 1).sum() > 0:
                labels[i] = 1
    return torch.Tensor(labels).reshape((len(g_out), 1))


def SGn_count(n=3):
    @ threshhold
    def SG_count_tmp(data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)
        labels = [0 for i in range(len(g_out))]
        for i in range(len(g_out)):
            if g_in[i].shape[0] > 0:
                two_hop_nbhd = torch.cat([g_in[int(j)] for j in g_in[i]]).unique()
                scatters = [len(list(set(g_out[int(k)].numpy()) & set(g_in[int(i)].numpy())-{i})) for k in two_hop_nbhd if k != i]
                if len(scatters) > 0:
                    labels[i] = sum([s>=n for s in scatters])
        return torch.Tensor(labels).reshape((len(g_out), 1))
    return SG_count_tmp

def BP_n_2_check(n=3):
    @ threshhold
    def BP_in_check_tmp(data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)
        labels = [0 for i in range(len(g_out))]
        for i in range(len(g_out)):
            if labels[i] == 1:
                continue
            if g_in[i].shape[0] > 0:
                two_hop_nbhd = torch.cat([g_out[int(j)] for j in g_in[i]]).unique()
                if len(two_hop_nbhd) > 0:
                    for k in two_hop_nbhd:
                        if k != i:
                            scatter = len(list( set(g_in[int(k)].numpy()) & set(g_in[int(i)].numpy())-{i} ))
                            if scatter >= n:
                                labels[i] = 1
                                labels[k] = 1
                                break
        return torch.Tensor(labels).reshape((len(g_out), 1))
    return BP_in_check_tmp

def BP_n_2_count(n=3):
    @ threshhold
    def BP_in_count_tmp(data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)
        labels = [0 for i in range(len(g_out))]
        for i in range(len(g_out)):
            if g_in[i].shape[0] > 0:
                two_hop_nbhd = torch.cat([g_out[int(j)] for j in g_in[i]]).unique()
                scatters = [len(list( set(g_in[int(k)].numpy()) & set(g_in[int(i)].numpy())-{i} )) for k in two_hop_nbhd if k != i]
                if len(scatters) > 0:
                    labels[i] = sum([s>=n for s in scatters])
        return torch.Tensor(labels).reshape((len(g_out), 1))
    return BP_in_count_tmp

@ threshhold
def SG_max(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_lists(edge_index, num_nodes)
    labels = [0 for i in range(len(g_out))]
    for i in range(len(g_out)):
        if g_in[i].shape[0] > 0:
            two_hop_nbhd = torch.cat([g_in[int(j)] for j in g_in[i]]).unique()
            scatters = [len(list(set(g_out[int(k)].numpy()) & set(g_in[int(i)].numpy())-{i})) for k in two_hop_nbhd if k != i]
            if len(scatters) > 0:
                labels[i] = max(scatters)
    return torch.Tensor(labels).reshape((len(g_out), 1))

@ threshhold
def weak_cc_size(data):
    gx = to_networkx(data, to_undirected=False)
    ccs = nx.weakly_connected_components(gx)
    y = torch.zeros((data.num_nodes, 1))
    for cc in ccs:
        for i in cc:
            y[i] = len(cc)
    return y.long()

@ threshhold
def strong_cc_size(data):
    gx = to_networkx(data, to_undirected=False)
    ccs = nx.strongly_connected_components(gx)
    y = torch.zeros((data.num_nodes, 1))
    for cc in ccs:
        for i in cc:
            y[i] = len(cc)
    return y.long()

@ threshhold
def cc_size(data):
    edge_index, edge_attr = data.edge_index, data.edge_attr
    edge_index = torch.sort(edge_index, dim=0)[0]
    edge_index, edge_attr = coalesce(edge_index, edge_attr)
    data2 = GraphData(data.x, edge_index, edge_attr, y=data.y)
    gx = to_networkx(data2, to_undirected=True)
    ccs = nx.connected_components(gx)
    y = torch.zeros((data.num_nodes, 1))
    for cc in ccs:
        for i in cc:
            y[i] = len(cc)
    return y.long()

def cc_check(data, min_size=4):
    gx = to_networkx(data, to_undirected=True)
    ccs = nx.connected_components(gx)
    big_cc_inds = [cc for cc in ccs if len(cc)>=min_size]
    label_inds = flatten(big_cc_inds)
    y = torch.zeros((data.num_nodes, 1))
    for i in label_inds:
        y[i] = 1
    return y.long()

@ threshhold
def gather_cc_check(data, min_size=4, min_fan_in=4, min_max_deg_out=2):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    g_in, g_out = to_adj_lists(edge_index, num_nodes)
    deg_out = torch.Tensor([len(g_out[node]) for node in range(len(g_out))]).reshape((len(g_out), 1)).long()
    deg_outs = [Counter(g_out[node]) for node in range(len(g_out))]
    max_deg_out = [max(cnt.values()) if cnt else 0 for cnt in deg_outs]
    deg_out_nodes = torch.Tensor([d>=min_max_deg_out for d in max_deg_out]).reshape((len(g_out), 1)).long()
    fan_in = torch.Tensor([len(set(g_in[node])) for node in range(len(g_in))]).reshape((len(g_in), 1)).long()
    fan_in_nodes = torch.Tensor([f>=min_fan_in for f in fan_in]).reshape((len(g_out), 1)).long()
    phishing_candidates = deg_out_nodes & fan_in_nodes
    phishing_candidate_inds = list(torch.where(phishing_candidates)[0].numpy())
    phishing_candidate_mothers = [[k for k,v in deg_outs[i].items() if v>=min_max_deg_out] for i in phishing_candidate_inds]
    mother_candidate_inds = list(set(flatten(phishing_candidate_mothers)))
    
    node_index = phishing_candidate_inds + mother_candidate_inds
    _edge_index, _edge_attr = subgraph(node_index, data.edge_index, data.edge_attr, relabel_nodes=False, num_nodes=data.num_nodes)
    _edge_index = torch.sort(_edge_index, dim=0)[0]
    _edge_index, _edge_attr = coalesce(_edge_index, _edge_attr)
    data2 = GraphData(data.x, _edge_index, _edge_attr, y=data.y)
    y = cc_check(data2, min_size=min_size)
    return (y & phishing_candidates).long()


### Manually Added Patterns 

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_gather_tree(nodes, total_graph_nodes, fanins, degouts):
    new_y = np.zeros(total_graph_nodes)
    fanins_plus = np.array(fanins) + 1
    phishing_inds = np.array([sum(fanins_plus[:i]) for i in range(1, len(fanins_plus)+1)])
    phishing_inds = phishing_inds - 1
    for i in phishing_inds:
        new_y[nodes[i]] = 1
        new_y[nodes[i+1]] = 1
    new_y = torch.Tensor(new_y).reshape((-1, 1)).long()
    
    fan_in_mask = [0 if i in phishing_inds or i>phishing_inds[-1] else 1 for i in range(len(nodes))]
    fanin_src = list(compress(nodes, fan_in_mask))
    fanin_dst = [[nodes[i]]*n for i,n in zip(phishing_inds, fanins)]
    fanin_dst = flatten(fanin_dst)

    deg_out_src = [[nodes[i]]*n for i,n in zip(phishing_inds, degouts)]
    deg_out_src = flatten(deg_out_src)
    deg_out_dst = [[nodes[i+1]]*n for i,n in zip(phishing_inds, degouts)]
    deg_out_dst = flatten(deg_out_dst)

    edge_src = torch.Tensor(fanin_src+deg_out_src).long().reshape((1,-1))
    edge_dst = torch.Tensor(fanin_dst+deg_out_dst).long().reshape((1,-1))
    edge_index = torch.cat([edge_src, edge_dst], dim=0)
    return edge_index, new_y

def add_decoy_parallel_edges(data, n=1, degout=5, poisson=True, scaling=3):
    if poisson:
        degouts = np.round(np.random.poisson(scaling*degout, (n))/scaling).astype(int) 
    else:
        degouts = [degout] * n
    logging.debug(f"degouts = {degouts}")

    node_list = list(range(data.y.shape[0]))
    num_pattern_nodes = 2*n
    nodes = np.random.choice(node_list, num_pattern_nodes, replace=False)
    new_edges = torch.Tensor(nodes).long().reshape((2, n))
    degouts = torch.Tensor(degouts).long()

    new_edge_index = new_edges.repeat_interleave(degouts, dim=1)
    max_time = int(data.timestamps.max())
    new_edge_attr = torch.randint(0, max_time, (new_edge_index.shape[1], 1))

    data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
    data.timestamps = torch.cat([data.timestamps, new_edge_attr.reshape(-1)])
    return data

def add_gather_tree(data, length=3, fanin=5, degout=5, poisson=True, scaling=3, num_decoy=0):
    num_decoy = length
    data = add_decoy_parallel_edges(data, n=num_decoy, degout=degout, poisson=poisson, scaling=scaling)
    if poisson:
        length = np.round(np.random.poisson(scaling*length)/scaling).astype(int)
        length = max(1, length)
        fanins = np.round(np.random.poisson(scaling*fanin, (length))/scaling).astype(int) 
        degouts = np.round(np.random.poisson(scaling*degout, (length))/scaling).astype(int) 
    else:
        fanins = [fanin] * length
        degouts = [degout] * length
    logging.debug(f"degouts = {degouts}")
    logging.debug(f"fanins = {fanins}")

    node_list = list(range(data.y.shape[0]))
    num_pattern_nodes = sum(fanins) + len(fanins) + 1
    nodes = np.random.choice(node_list, num_pattern_nodes, replace=False)

    new_edge_index, new_y = make_gather_tree(nodes, len(node_list), fanins, degouts)
    max_time = int(data.timestamps.max())
    new_edge_attr = torch.randint(0, max_time, (new_edge_index.shape[1], 1))

    data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
    data.timestamps = torch.cat([data.timestamps, new_edge_attr.reshape(-1)])
    data.y = data.y | new_y
    return data

### END 


class GraphSimulator(object):
    def __init__(
        self, num_nodes: int, avg_degree: int = None, num_edges: int = None, max_time: int = None, 
        network_type: str = 'type1', readout: str = 'node', node_feats=False, bidirectional=False,
        delta=None, num_graphs: int = None, generator: str = 'chordal'
        ):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        # some default values
        self.num_edges = int(num_nodes * avg_degree / 2) if num_edges is None else num_edges
        self.avg_degree = (num_edges / num_nodes) * 2 if avg_degree is None else avg_degree
        self.max_time = num_nodes if max_time is None else max_time
        self.network_type = network_type
        self.readout = readout
        self.node_feats = node_feats
        self.bidirectional = bidirectional
        self.delta = delta
        if generator == 'barabasi':
            self.generator = self.barabasi_albert
        elif generator == 'chordal':
            self.generator = self.random_chordal_graph
        elif generator == 'random':
            self.generator = self.random_generator

    def random_chordal_graph(self, i=0):
        avg_degree = self.avg_degree
        if i>0: avg_degree = np.random.poisson(avg_degree)
        logging.info(f"Component {i} avg_degree = {avg_degree}")
        delta = self.delta if self.delta is not None else avg_degree * 0.5
        num_edges = int(self.num_nodes * avg_degree / 2)
        src = torch.randint(0, self.num_nodes, (1, 2*num_edges))
        dst = torch.normal(src.float(), delta).round().long() % self.num_nodes
        edge_index = torch.cat([src, dst], dim=0)
        edge_index = edge_index[:,torch.where(edge_index[0] != edge_index[1])[0]]
        while edge_index.shape[1]<num_edges:
            src = torch.randint(0, self.num_nodes, (1, 2*num_edges))
            dst = torch.normal(src.float(), delta).round().long()
            edge_index_tmp = torch.cat([src, dst], dim=0)
            edge_index_tmp = edge_index_tmp[:,torch.where(edge_index_tmp[0] != edge_index_tmp[1])[0]]
            edge_index = torch.cat([edge_index, edge_index_tmp], dim=1)
        edge_index = edge_index[:,:num_edges]
        return edge_index

    def erdos_renyi_graph(self, i=0):
        edge_index = torch.randint(0, self.num_nodes, (2, 2*self.num_edges))
        edge_index = edge_index[:,torch.where(edge_index[0] != edge_index[1])[0]]
        while edge_index.shape[1]<self.num_edges:
            edge_index_tmp = torch.randint(0, self.num_nodes, (2, 2*self.num_edges))
            edge_index_tmp = edge_index_tmp[:,torch.where(edge_index_tmp[0] != edge_index_tmp[1])[0]]
            edge_index = torch.cat([edge_index, edge_index_tmp], dim=1)
        edge_index = edge_index[:,:self.num_edges]
        return edge_index

    def barabasi_albert(self, i=0):
        n = self.num_nodes
        m = round(self.avg_degree / 2)
        if i>0: m = np.random.poisson(m)
        logging.info(f"Component {i} avg_degree = {m}")
        nx_graph = nx.barabasi_albert_graph(n, m)
        data = from_networkx(nx_graph)
        edge_index = data.edge_index
        flip = torch.randint(2,(1, edge_index.shape[1]))
        edge_index = torch.where(flip.bool(), edge_index.flipud(), edge_index)
        return edge_index

    def random_generator(self, i=0):
        generator = np.random.choice([self.random_chordal_graph, self.barabasi_albert])
        logging.info(f"generator = {generator}")
        return generator(i)

    def generate_edge_indices(self):
        edge_index = torch.zeros((2,0)).long()
        max_node = torch.Tensor([0]).long()
        for i in range(self.num_graphs):
            edge_index_tmp = self.generator(i)
            edge_index = torch.cat((edge_index, edge_index_tmp + max_node), dim=1)
            max_node = edge_index.max()+1
        return edge_index

    def generate_pytorch_graph(self):
        edge_index = self.generate_edge_indices()
        total_nodes = int(edge_index.long().max())+1
        total_edges = int(edge_index.shape[1])
        timestamps = torch.randint(0, self.max_time, (total_edges, 1))
        timestamps2 = torch.randint(0, self.max_time, (total_edges, 1))
        if self.bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flipud()], dim=1)
            timestamps = torch.cat([timestamps, timestamps], dim=0)
            timestamps2 = torch.cat([timestamps2, timestamps2], dim=0)
        # edge_attr = timestamps.float()
        edge_attr = torch.cat([timestamps.reshape((-1,1)).float(), timestamps2.reshape((-1,1)).float()], dim=1)
        num_samples = total_nodes if self.readout == 'node' else total_edges
        x = torch.ones((total_nodes, 1))
        y = torch.zeros((num_samples, 1)).long()
        return GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, readout=self.readout, num_nodes=total_nodes)

functions = {
    'deg_in':           {"set_y": deg_in,           "default": 6,  "delta": None},
    'deg_out':          {"set_y": deg_out,          "default": 6,  "delta": None},
    'fan_in':           {"set_y": fan_in,           "default": 6,  "delta": None},
    'fan_out':          {"set_y": fan_out,          "default": 6,  "delta": None},
    'max_fan_in':       {"set_y": max_fan_in,       "default": 6,  "delta": None},
    'max_deg_in':       {"set_y": max_deg_in,       "default": 6,  "delta": None},
    'ratio_in':         {"set_y": ratio_in,         "default": 6,  "delta": None},
    'ratio_out':        {"set_y": ratio_out,        "default": 6,  "delta": None},
    'timevar_in':       {"set_y": timevar_in,       "default": 6,  "delta": None},
    'timevar_out':      {"set_y": timevar_out,      "default": 6,  "delta": None},
    'e_time_mod7':      {"set_y": e_time_mod7,      "default": 6,  "delta": None},
    'deg_in_check':     {"set_y": deg_in,           "default": 6,  "delta": None,     "thresh": 'median'},
    'deg_out_check':    {"set_y": deg_out,          "default": 6,  "delta": None,     "thresh": 'median'},
    'fan_in_check':     {"set_y": fan_in,           "default": 6,  "delta": None,     "thresh": 'median'},
    'fan_out_check':    {"set_y": fan_out,          "default": 6,  "delta": None,     "thresh": 'median'},
    'max_fan_in_check': {"set_y": max_fan_in,       "default": 6,  "delta": None,     "thresh": 'median'},
    'max_deg_in_check': {"set_y": max_deg_in,       "default": 6,  "delta": None,     "thresh": 'median'},
    'ratio_in_check':   {"set_y": ratio_in,         "default": 6,  "delta": None,     "thresh": 'median'},
    'ratio_out_check':  {"set_y": ratio_out,        "default": 6,  "delta": None,     "thresh": 'median'},
    'timevar_in_check': {"set_y": timevar_in,       "default": 6,  "delta": None,     "thresh": 'median'},
    'timevar_out_check':{"set_y": timevar_out,      "default": 6,  "delta": None,     "thresh": 'median'},
    'e_time_mod7_check':{"set_y": e_time_mod7,      "default": 6,  "delta": None,     "thresh": 0.5},
    'C2_check':         {"set_y": Cn_check(2),      "default": 6,  "delta": None},
    'C3_check':         {"set_y": Cn_check(3),      "default": 6,  "delta": None},
    'C4_check':         {"set_y": Cn_check(4),      "default": 6,  "delta": lambda x: 3**(x-4)-0.5     }, # (5,2.5)
    'C5_check':         {"set_y": Cn_check(5),      "default": 6,  "delta": lambda x: 4**(x-4)-0.5     }, # (5,2.5)
    'C6_check':         {"set_y": Cn_check(6),      "default": 6,  "delta": lambda x: 5**(x-4)-0.5     }, # (5,2.5)
    'C2_count':         {"set_y": C2_count,         "default": 6,  "delta": None},
    'C3_count':         {"set_y": C3_count,         "default": 6,  "delta": None},
    'C4_count':         {"set_y": C4_count,         "default": 6,  "delta": None},
    'deg_in_timespan':  {"set_y": deg_in_timespan,  "default": 6,  "delta": None},
    'deg_in_timespan_check':{"set_y": deg_in_timespan,  "default": 6,  "delta": None,     "thresh": 'median'}, # default avg_degree up to here!
    'K3_check':         {"set_y": Kn_check(3),      "default": 4,  "delta": lambda x: x**2 - 10        }, # (4,6), (6,26), (8,54)
    'K4_check':         {"set_y": Kn_check(4),      "default": 8,  "delta": lambda x: (2.75 * x) - 18  }, # (8,4)
    'K5_check':         {"set_y": Kn_check(5),      "default":12,  "delta": lambda x: 1.5*x - 15       }, # (12,3)
    'K6_check':         {"set_y": Kn_check(6),      "default":20,  "delta": lambda x: x - 14           }, # (20,6)
    'K7_check':         {"set_y": Kn_check(7),      "default":26,  "delta": lambda x: 0.9*x-18         }, # (26,5.4)
    'K3_count':         {"set_y": Kn_count(3),      "default": 4,  "delta": lambda x: x                }, # (4,4)
    'K4_count':         {"set_y": Kn_count(4),      "default": 8,  "delta": lambda x: 1.5*x - 8        }, # (8,4)
    'K5_count':         {"set_y": Kn_count(5),      "default":14,  "delta": lambda x: 0.5*x - 3        }, # (14,4)
    'K6_count':         {"set_y": Kn_count(6),      "default":20,  "delta": lambda x: 4                }, # (20,4)
    'K7_count':         {"set_y": Kn_count(7),      "default":26,  "delta": lambda x: 4                }, # (26,4)
    # 'SG1_check':        {"set_y": SGn_check(1),     "default":2,   "delta": lambda x: 8                }, # (2,8)
    # 'SG2_check':        {"set_y": SGn_check(2),     "default":4,   "delta": lambda x: 8                }, # (7,8)
    # 'SG3_check':        {"set_y": SGn_check(3),     "default":9,   "delta": lambda x: 8                }, # (12,8)
    # 'SG4_check':        {"set_y": SGn_check(4),     "default":11,  "delta": lambda x: 6                }, # (15,6)
    # 'SG_max':           {"set_y": SG_max,           "default":2,   "delta": lambda x: 8                },
    'SG1_check':        {"set_y": SGn_check(1),     "default":2,   "delta": lambda x: 8                }, # (2,8)
    'SG2_check':        {"set_y": SG2_check,        "default":7,   "delta": lambda x: 8                }, # (7,8)
    'SG3_check':        {"set_y": SGn_check(3),     "default":12,  "delta": lambda x: 8                }, # (12,8)
    'SG4_check':        {"set_y": SGn_check(4),     "default":15,  "delta": lambda x: 6                }, # (15,6)
    'SG1_count':        {"set_y": SGn_count(1),     "default":2,   "delta": lambda x: 8                }, # (2,8)
    'SG2_count':        {"set_y": SGn_count(2),     "default":7,   "delta": lambda x: 8                }, # (7,8)
    'SG3_count':        {"set_y": SGn_count(3),     "default":12,  "delta": lambda x: 8                }, # (12,8)
    'SG4_count':        {"set_y": SGn_count(4),     "default":15,  "delta": lambda x: 6                }, # (15,6)
    'BP1_check':        {"set_y": BP_n_2_check(1),  "default":2,   "delta": lambda x: 8                }, # (2,8)
    'BP2_check':        {"set_y": BP2_check,        "default":7,   "delta": lambda x: 8                }, # (7,8)
    'BP3_check':        {"set_y": BP_n_2_check(3),  "default":12,  "delta": lambda x: 8                }, # (12,8)
    'BP4_check':        {"set_y": BP_n_2_check(4),  "default":15,  "delta": lambda x: 6                }, # (15,6)
    'BP1_count':        {"set_y": BP_n_2_count(1),  "default":2,   "delta": lambda x: 8                }, # (2,8)
    'BP2_count':        {"set_y": BP_n_2_count(2),  "default":7,   "delta": lambda x: 8                }, # (7,8)
    'BP3_count':        {"set_y": BP_n_2_count(3),  "default":12,  "delta": lambda x: 8                }, # (12,8)
    'BP4_count':        {"set_y": BP_n_2_count(4),  "default":15,  "delta": lambda x: 6                }, # (15,6)
    'SG_max':           {"set_y": SG_max,           "default":4,   "delta": lambda x: 8                },
    'SG_max_check':     {"set_y": SG_max,           "default":8,   "delta": lambda x: x,     "thresh": 'median'},
    'gather_cc_check':  {"set_y": gather_cc_check,  "default":12,  "delta": lambda x: 8                },
}

add_pattern_functions = {
    'gather_tree':      {"set_y": add_gather_tree,             "default": 6,  "delta": lambda x: 8},
}

def get_gnn_data_from_simulator(config, args, num_edges=None, max_time=None, network_type='type1', readout='node', node_feats=False):
    if sum([y in add_pattern_functions for y in args.y_list]) > 0:
        assert len(args.y_list) == 1, f"Only one manually added pattern allowed. {args.y_list} is too long."
        manual_builder = True
        sim_params = [add_pattern_functions[y] for y in args.y_list]
    else:
        manual_builder = False
        sim_params = [functions[y] for y in args.y_list]
    set_y =             [sim_param["set_y"] for sim_param in sim_params]
    default_degrees =   [sim_param["default"] for sim_param in sim_params]
    delta_fcts =        [sim_param["delta"] for sim_param in sim_params if sim_param["delta"] is not None]
    threshes =          [sim_param["thresh"] if "thresh" in sim_param else None for sim_param in sim_params]
    default_degree = min(default_degrees)

    # Set the average degree. Priority list: args.sim_avg_degree, then default_degree, then 6
    if args.sim_avg_degree is not None:
        avg_degree = args.sim_avg_degree
        logging.warning(f"Data might be very imbalanced when using custom avg_degree. Set avg_degree = None to use default avg_degree values.")
    elif default_degree is not None:
        avg_degree = (default_degree / 2) + 1  if args.bidirectional_simulator else default_degree
    else:
        avg_degree = 6
    # Set delta (i.e., the standard deviation of the distance of the destination node from the source node) for all edges. 
    # If delta is small, then connectivity is very localized in the generated graph.
    if args.sim_delta is not None:
        delta = args.sim_delta
        logging.warning(f"Data might be very imbalanced when using custom delta. Set delta = None to use default delta values.")
    elif delta_fcts:
        deltas = [delta_fct(avg_degree) for delta_fct in delta_fcts]
        delta = np.mean(deltas)
    else:
        delta = 6
    logging.info(f"avg_degree, delta = {avg_degree}, {delta}")
    # Set up graph simulator with parameters
    graph_simulator = GraphSimulator(
        num_nodes=args.sim_num_nodes, avg_degree=avg_degree, num_edges=num_edges, max_time=max_time, 
        network_type=network_type, readout=readout, node_feats=node_feats, bidirectional=args.bidirectional_simulator,
        delta=delta, num_graphs=args.sim_num_graphs, generator=args.sim_generator
        )
    # Generate train, valid and test graphs
    tr_data = graph_simulator.generate_pytorch_graph()
    val_data = graph_simulator.generate_pytorch_graph()
    te_data = graph_simulator.generate_pytorch_graph()
    # Get threshholds
    threshholds = []
    for func, thresh in zip(set_y, threshes):
        if thresh is None:
            threshholds.append(thresh)
        elif thresh == 'median':
            threshholds.append(get_median(tr_data, func))
        else:
            threshholds.append(thresh)
    logging.info(f"threshholds = {threshholds}")
    # Process datasets
    for data in [tr_data, val_data, te_data]:
        if manual_builder:
            # Add synthetic data and labels
            logging.info(f"using graph builder for {args.y_list[0]}.")
            builder = set_y[0]
            for _ in range(args.sim_num_patterns):
                data = builder(data, scaling=2)
            logging.info(f"Done building.")
        else:
            # Calculate sythetic labels
            data.set_y(set_y, threshholds)
    # Write label totals to file
    with open(f'{args.log_dir}/y_sums.csv', 'w') as outfile:
        outfile.write(','.join(args.y_list))
        outfile.write(',total\n')
        for data in [tr_data, val_data, te_data]:
            outfile.write(','.join([str(i) for i in list(sum(data.y).numpy())]))
            outfile.write(f',{len(data.y)}\n')
    return tr_data, val_data, te_data


def apply_simulator(data_list, args, return_threshholds=False):
    tr_data = data_list[0]
    if sum([y in add_pattern_functions for y in args.y_list]) > 0:
        assert len(args.y_list) == 1, f"Only one manually added pattern allowed. {args.y_list} is too long."
        manual_builder = True
        sim_params = [add_pattern_functions[y] for y in args.y_list]
    else:
        manual_builder = False
        sim_params = [functions[y] for y in args.y_list]
    set_y =             [sim_param["set_y"] for sim_param in sim_params]
    threshes =          [sim_param["thresh"] if "thresh" in sim_param else None for sim_param in sim_params]
    # Get threshholds
    threshholds = []
    for func, thresh in zip(set_y, threshes):
        if thresh is None:
            threshholds.append(thresh)
        elif thresh == 'median':
            threshholds.append(get_median(tr_data, func))
        else:
            threshholds.append(thresh)
    logging.info(f"threshholds = {threshholds}")
    # Process datasets
    data_out = []
    for data in data_list:
        if manual_builder:
            # Add synthetic data and labels
            logging.info(f"using graph builder for {args.y_list[0]}.")
            builder = set_y[0]
            for _ in range(args.sim_num_patterns):
                data = builder(data, scaling=2)
            logging.info(f"Done building.")
        else:
            # Calculate sythetic labels
            data.set_y(set_y, threshholds)
        data_out.append(data)
    if return_threshholds: data_out.append(threshholds)
    return data_out
