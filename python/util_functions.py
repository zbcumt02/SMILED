import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx


class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    print('successful sampling positive and negative links for train and test')
    return train_pos, train_neg, test_pos, test_neg


def sample_neg_mll_small(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, row_range=None,
                         col_range=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # pos at specific interval
    mask = (row >= row_range[0]) & (row < row_range[1]) & (col >= col_range[0]) & (col < col_range[1])
    row, col = row[mask], col[mask]

    num_rows = len(row)
    num_test_rows = int(math.ceil(num_rows * test_ratio))

    test_row_indices = np.random.choice(row, size=num_test_rows, replace=False)
    test_mask = np.isin(row, test_row_indices)

    test_pos = (row[test_mask], col[test_mask])

    train_mask = ~test_mask
    train_pos = (row[train_mask], col[train_mask])

    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])

    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    print('sampling negative links')

    # generating test_neg: only screening in specific interval
    test_neg = ([], [])
    while len(test_neg[0]) < test_num:
        i = random.randint(row_range[0], row_range[1] - 1)
        j = random.randint(col_range[0], col_range[1] - 1)
        if i < j and net[i, j] == 0 and (i, j) not in zip(test_neg[0], test_neg[1]):
            test_neg[0].append(i)
            test_neg[1].append(j)

    # generating pos_neg: only screening in others
    train_neg = ([], [])
    while len(train_neg[0]) < train_num:
        i = random.randint(row_range[0], row_range[1] - 1)
        j = random.randint(col_range[0], col_range[1] - 1)
        if (i < j and net[i, j] == 0 and (i, j) not in zip(train_neg[0], train_neg[1])
                and (i, j) not in zip(test_neg[0], test_neg[1])):
            train_neg[0].append(i)
            train_neg[1].append(j)

    print('successful sampling positive and negative links for train pos with %d'
          ' and test pos with %d, train neg with %d and test neg with %d'
          % (train_num, test_num, len(train_neg[0]), len(test_neg[0])))
    return train_pos, train_neg, test_pos, test_neg


def sample_neg_mll_small_dual(net, max_train_num=None, row_range=None, col_range=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)
    mask = (row >= row_range[0]) & (row < row_range[1]) & (col >= col_range[0]) & (col < col_range[1])
    row, col = row[mask], col[mask]

    mask = np.isin(row, row)
    pos = (row[mask], col[mask])

    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(pos[0]))[:max_train_num]
        pos = (pos[0][perm], pos[1][perm])

    # sample negative links for train/test
    print('sampling negative links')

    neg = ([], [])
    for i in range(row_range[0], row_range[1]):
        for j in range(col_range[0], col_range[1]):
            if net[i, j] == 0 and (i, j) not in zip(neg[0], neg[1]):
                neg[0].append(i)
                neg[1].append(j)

    print('successful sampling positive and negative links for pos with %d and neg with %d'
          % (len(pos[0]), len(neg[0])))
    return pos, neg


def cluster_sample(pos_pairs, neg_pairs, node_attributes, n_clusters_pos=0.1, n_clusters_neg=0.1, samples_per_cluster=5):
    from sklearn.cluster import KMeans
    import numpy as np

    def sample_from_pairs(pairs, is_pos=True):
        if isinstance(pairs[0], list):
            pairs_array = (np.array(pairs[0]), np.array(pairs[1]))
        else:
            pairs_array = pairs

        if len(pairs_array[0]) == 0 or pairs_array[0].size == 0:
            return pairs_array

        # Transforming sample pairs into features: feature stitching using nodes at both ends
        feats = []
        for u, v in zip(pairs_array[0], pairs_array[1]):
            # Ensure that the index is an integer.
            u_idx, v_idx = int(u), int(v)
            # Ensure that the index is not out of bounds
            if u_idx >= len(node_attributes) or v_idx >= len(node_attributes):
                # If the index is out of bounds, use the zero vector instead
                feat = np.zeros(node_attributes.shape[1] * 2)
            else:
                feat_u = node_attributes[u_idx]
                feat_v = node_attributes[v_idx]
                # Ensure that the feature is a one-dimensional array
                feat_u = feat_u.flatten() if hasattr(feat_u, 'flatten') else feat_u
                feat_v = feat_v.flatten() if hasattr(feat_v, 'flatten') else feat_v
                feat = np.concatenate([feat_u, feat_v])
            feats.append(feat)

        feats = np.array(feats)
        if is_pos:
            k = int(len(pos_pairs[0]) * n_clusters_pos)
        else:
            k = int(len(neg_pairs[0]) * n_clusters_neg)

        # clustering
        n_clusters_actual = min(k, len(feats))
        if n_clusters_actual <= 1:
            n_samples = min(samples_per_cluster, len(feats))
            indices = np.random.choice(len(feats), n_samples, replace=False)
        else:
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feats)
            indices = []
            for cluster_id in range(n_clusters_actual):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # choose 'samples_per_cluster' samples per cluster
                    if len(cluster_indices) <= samples_per_cluster:
                        indices.extend(cluster_indices.tolist())
                    else:
                        distances = np.linalg.norm(feats[cluster_indices] - kmeans.cluster_centers_[cluster_id], axis=1)
                        closest = np.argsort(distances)[:samples_per_cluster]
                        indices.extend(cluster_indices[closest].tolist())

        # enough samples
        if len(indices) == 0:
            # use all
            indices = list(range(len(pairs_array[0])))

        # Return the sampled sample pairs in their original format
        if isinstance(pairs[0], list):
            # list
            sampled = (pairs_array[0][indices].tolist(), pairs_array[1][indices].tolist())
        else:
            # numpy
            sampled = (pairs_array[0][indices], pairs_array[1][indices])
        return sampled

    sampled_pos = sample_from_pairs(pos_pairs, is_pos=True)

    sampled_neg = sample_from_pairs(neg_pairs, is_pos=False)

    return sampled_pos, sampled_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}

    def helper(A, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information)
                                                   for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def links2subgraphs_dual(A, pos, neg, h=1, max_nodes_per_hop=None, node_information=None):
    # extract enclosing subgraphs
    max_n_label = {'value': 0}

    def helper(A, links, g_label):
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information)
                                                   for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label_value = 0
        for _, n_labels, _ in results:
            if n_labels:
                max_n_label_value = max(max(n_labels), max_n_label_value)
        max_n_label['value'] = max(max_n_label_value, max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    graphs = helper(A, pos, 1) + helper(A, neg, 0)
    print(max_n_label)
    return graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    # g = nx.from_scipy_sparse_matrix(subgraph)
    g = nx.from_scipy_sparse_array(subgraph)
    # remove link between target nodes
    # print(len(nodes))
    # print(ind, nodes)

    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    # print(g, labels, labels.tolist(), features)
    return g, labels.tolist(), features

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    # replace
    # when print(dist_to_0), will get error, it has lots of inf
    # dist_to_0 = np.nan_to_num(dist_to_0, nan=0, posinf=100, neginf=-100)
    # dist_to_1 = np.nan_to_num(dist_to_1, nan=0, posinf=100, neginf=-100)

    d = (dist_to_0 + dist_to_1).astype(int)

    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)

def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def gnn_to_line(batch_graph, max_n_label):
    start = time.time()
    pool = mp.Pool(16)
    #pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_line_worker, [(graph, max_n_label) for graph in batch_graph])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g in results]
    return g_list

def parallel_line_worker(x):
    return to_line(*x)

def to_line(graph, max_n_label):
    edges = graph.edge_pairs
    edge_feas = edge_fea(graph, max_n_label)/2
    edges, feas = to_undirect(edges, edge_feas)
    edges = torch.tensor(edges)
    data = Data(edge_index=edges, edge_attr=feas)
    data.num_nodes = graph.num_nodes
    data = LineGraph()(data)
    data.num_nodes = graph.num_edges
    data['y'] = torch.tensor([graph.label])
    return data


def to_edgepairs(graph):
    x, y = zip(*graph.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y
    edge_pairs = edge_pairs.flatten()
    return edge_pairs


def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)

        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges

        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1, 1)
    node_tag.scatter_(1, tags, 1)
    node_attr = torch.tensor(graph.node_features, dtype=torch.float32)
    node_tag = torch.cat((node_tag, node_attr), dim=1)
    return node_tag

def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1, 2))
    sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)
    fea_s = edge_fea[sr[0, :], :]
    fea_s = fea_s.repeat(2, 1)
    fea_r = edge_fea[sr[1, :], :]
    fea_r = fea_r.repeat(2, 1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body


def single_line(batch_graphs):
    pbar = tqdm(batch_graphs, unit='iteration')
    graphs = []
    for graph in pbar:
        #line_graph, labels = to_line(graph, graph.node_tags)
        line_test(graph, graph.node_tags)
        #graphs.append(line_graph)
    return graphs


def line_test(graph, label):
    edges = graph.edge_pairs
    edges = to_undirect2(edges)
    feas = edge_fea2(label, edges)
    data = Data(edge_index=torch.tensor(edges), edge_attr=feas.T)
    data = LineGraph()(data)
    elist = data['edge_index'].numpy()
    #elist = [(elist[0][i], elist[1][i]) for i in range(len(elist[0]))]
    #nx_graph = nx.Graph()
    #nx_graph.add_edges_from(elist)
    #return nx_graph, data['x'].numpy()
    #return nx
    

def edge_fea2(labels, edges):
    feas = []
    for i in range(edges.shape[1]):
        fea = [labels[edges[0][i]], labels[edges[1][i]]]
        fea.sort()
        feas.append(fea)
    feas = np.reshape(feas, [-1, 2])
    feas = np.array([feas[:,0], feas[:,1]], dtype=np.float32)
    return torch.tensor(feas/2)


def to_undirect2(edges):
    edges = np.reshape(edges, (-1, 2))
    sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)
    rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
    target_edge = np.array([[0, 1], [1, 0]])
    return np.concatenate([target_edge, sr, rs], axis=1)


def label_wise_instance_filtering(targets_all, attributes_all, label_idx,
                                  filter_method='isolation_forest',
                                  contamination=0.1,
                                  adaptive_threshold=True):
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.cluster import DBSCAN

    # all positive sample
    pos_mask = targets_all[:, label_idx] == 1
    pos_indices = np.where(pos_mask)[0]

    if len(pos_indices) == 0:
        return []

    pos_features = attributes_all[pos_indices]

    # adaptive threshold
    if adaptive_threshold and len(pos_indices) < 20:
        adjusted_contamination = 0.05
    elif adaptive_threshold and len(pos_indices) < 50:
        adjusted_contamination = contamination * 0.5
    else:
        adjusted_contamination = contamination

    # denoise
    if filter_method == 'isolation_forest':
        clf = IsolationForest(contamination=adjusted_contamination,
                              random_state=42, n_estimators=100)
        preds = clf.fit_predict(pos_features)
        clean_mask = preds == 1

    elif filter_method == 'lof':
        clf = LocalOutlierFactor(contamination=adjusted_contamination,
                                 novelty=False)
        preds = clf.fit_predict(pos_features)
        clean_mask = preds == 1

    elif filter_method == 'elliptic_envelope':
        try:
            clf = EllipticEnvelope(contamination=adjusted_contamination,
                                   random_state=42)
            preds = clf.fit_predict(pos_features)
            clean_mask = preds == 1
        except:
            clf = IsolationForest(contamination=adjusted_contamination,
                                  random_state=42)
            preds = clf.fit_predict(pos_features)
            clean_mask = preds == 1

    elif filter_method == 'dbscan':
        clf = DBSCAN(eps=0.5, min_samples=5)
        preds = clf.fit_predict(pos_features)
        # -1: noise
        clean_mask = preds != -1
        if clean_mask.sum() / len(clean_mask) < 0.3:
            clf = DBSCAN(eps=0.8, min_samples=3)
            preds = clf.fit_predict(pos_features)
            clean_mask = preds != -1

    else:
        clf = IsolationForest(contamination=adjusted_contamination,
                              random_state=42)
        preds = clf.fit_predict(pos_features)
        clean_mask = preds == 1

    # clean sample indices
    clean_indices = pos_indices[clean_mask].tolist()

    print(f"Label {label_idx}: Filtered {len(pos_indices)} -> {len(clean_indices)} "
          f"positive instances (kept {len(clean_indices) / max(1, len(pos_indices)) * 100:.1f}%)")

    return clean_indices


def build_clean_sample_pools(targets_all, attributes_all,
                             filter_params=None):
    if filter_params is None:
        filter_params = {
            'filter_method': 'isolation_forest',
            'contamination': 0.1,
            'adaptive_threshold': True
        }

    num_labels = targets_all.shape[1]
    clean_pools = {}

    print(f"Building clean sample pools for {num_labels} labels on training set...")

    for label_idx in range(num_labels):
        clean_indices = label_wise_instance_filtering(
            targets_all, attributes_all, label_idx,
            filter_method=filter_params.get('filter_method', 'isolation_forest'),
            contamination=filter_params.get('contamination', 0.1),
            adaptive_threshold=filter_params.get('adaptive_threshold', True)
        )
        clean_pools[label_idx] = clean_indices

    total_original = np.sum(targets_all == 1)
    total_clean = sum(len(indices) for indices in clean_pools.values())

    print(f"Total positive instances: {total_original} -> {total_clean} "
          f"(reduction: {100 * (1 - total_clean / max(1, total_original)):.1f}%)")

    return clean_pools  # {label_idx: [clean_instance_indices]}


def sample_clean_pairs_dual(train_net, clean_pools, label_idx, targets_train,
                            max_train_num=None, row_range=None, col_range=None):
    import numpy as np
    import random
    import scipy.sparse as ssp

    # 1. pos
    pos_instances = clean_pools[label_idx]

    if len(pos_instances) == 0:
        print(f"Label {label_idx}: No clean positive instances available")
        return ([], []), ([], [])

    # 2. neg
    neg_candidates = []
    num_class = len(targets_train[0])
    for other_label, instances in clean_pools.items():
        if other_label == label_idx:
            continue

        for inst_idx in instances:
            if targets_train[inst_idx, label_idx] != 1:
                # others
                if np.sum(targets_train[inst_idx, :]) > 1 - num_class + 1:
                    neg_candidates.append(inst_idx)

    neg_candidates = list(set(neg_candidates))

    print(f"Label {label_idx}: {len(pos_instances)} clean positives, {len(neg_candidates)} clean negative candidates")

    train_net_triu = ssp.triu(train_net, k=1)
    row, col, _ = ssp.find(train_net_triu)

    mask = (row >= row_range[0]) & (row < row_range[1]) & (col >= col_range[0]) & (col < col_range[1])
    row, col = row[mask], col[mask]

    clean_pos_pairs = []
    for r, c in zip(row, col):
        if r in pos_instances and c == col_range[0]:
            clean_pos_pairs.append((r, c))

    if clean_pos_pairs:
        pos_rows, pos_cols = zip(*clean_pos_pairs)
        pos = (list(pos_rows), list(pos_cols))
    else:
        pos = ([], [])

    clean_neg_pairs = []

    for i in range(row_range[0], row_range[1]):
        for j in range(col_range[0], col_range[1]):
            if train_net[i, j] == 0:
                if i in neg_candidates and j == col_range[0]:
                    clean_neg_pairs.append((i, j))

    if len(clean_neg_pairs) > 0:
        if max_train_num is not None and len(clean_neg_pairs) > max_train_num:
            clean_neg_pairs = random.sample(clean_neg_pairs, max_train_num)

        neg_rows, neg_cols = zip(*clean_neg_pairs)
        neg = (list(neg_rows), list(neg_cols))
    else:
        neg = ([], [])

    if max_train_num is not None and len(pos[0]) > max_train_num:
        indices = np.random.choice(len(pos[0]), max_train_num, replace=False)
        pos = (np.array(pos[0])[indices].tolist(),
               np.array(pos[1])[indices].tolist())

    print(f"Label {label_idx}: Sampled {len(pos[0])} positive, {len(neg[0])} negative pairs from clean pool")

    return pos, neg


def create_co_teaching_data_loader(train_graphs, max_n_label, batch_size=50,
                                   keep_indices=None, remove_indices=None):
    """
    Create a data loader for co-teaching
    Args:
        train_graphs: List of original training graphs
        max_n_label: Maximum node label
        batch_size: Batch size
        keep_indices: Sample indices to be retained
        remove_indices: Sample indices to be removed (noise samples)

    Returns:
        DataLoader: The data loader after processing
    """
    from torch_geometric.data import DataLoader

    if keep_indices is not None:
        selected_graphs = [train_graphs[i] for i in keep_indices
                           if i < len(train_graphs)]
    elif remove_indices is not None:
        selected_graphs = []
        remove_set = set(remove_indices)
        for i, graph in enumerate(train_graphs):
            if i not in remove_set:
                selected_graphs.append(graph)
    else:
        selected_graphs = train_graphs

    if selected_graphs:
        selected_lines = to_linegraphs(selected_graphs, max_n_label)
        loader = DataLoader(selected_lines, batch_size=batch_size, shuffle=True)
    else:
        selected_lines = []
        loader = DataLoader(selected_lines, batch_size=batch_size, shuffle=True)

    return loader, len(selected_graphs)


def compute_sample_losses(model, data_loader, net_id='a', device='cuda'):
    model.eval()
    sample_losses = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)

            if net_id == 'a':
                logits, loss, _, _, _, _ = model.network_a(batch)
            else:
                logits, loss, _, _, _, _ = model.network_b(batch)

            # Get the loss of each sample
            # (to simplify things here, you should actually
            # calculate the cross-entropy loss of each sample)
            batch_size = len(batch.y)
            for i in range(batch_size):
                global_idx = batch_idx * data_loader.batch_size + i
                sample_losses[global_idx] = loss.item() / batch_size

    return sample_losses


def select_noise_samples_by_loss(disagreement_indices, losses_a, losses_b,
                                 forget_rate=0.3, total_samples=1000):
    if not disagreement_indices:
        return []

    avg_losses = []
    for idx in disagreement_indices:
        loss_a = losses_a.get(idx, 0)
        loss_b = losses_b.get(idx, 0)
        avg_loss = (loss_a + loss_b) / 2
        avg_losses.append((idx, avg_loss))

    avg_losses.sort(key=lambda x: x[1], reverse=True)
    n_to_select = int(forget_rate * total_samples)
    n_to_select = min(n_to_select, len(avg_losses))

    noise_indices = [idx for idx, _ in avg_losses[:n_to_select]]

    print(f"Selected {len(noise_indices)} high-loss samples as potential noise "
          f"(forget rate: {forget_rate:.1%})")

    return noise_indices
