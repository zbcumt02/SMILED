import numpy as np
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def generate_net():
    # 1. loading .mat file
    dataset = 'yeast'
    mat = scipy.io.loadmat('../data/%s/finaldata.mat' % dataset)
    for train_or_test in range(2):
        if train_or_test == 0:
            train_or_test = 'train'
            Xapp = mat['Xapp']  # shape: (num_ins, num_fea)
            Yapp = mat['Yapp']  # shape: (num_ins, num_class)
        else:
            train_or_test = 'test'
            Xapp = mat['Xgen']
            Yapp = mat['Ygen']

        trans_ratio = 0.001  # ratio of ins_ins change to 1 comparing with ins_label

        # 2. initialize
        num_ins, num_class, num_fea = int(Yapp.shape[0]), int(Yapp.shape[1]), int(Xapp.shape[1])
        num_all = num_ins + num_class
        adj_matrix = np.zeros((num_all, num_all), dtype=np.int8)

        # 3. instance-label link construction
        # Yapp: 1/-1
        instance_label_links = (Yapp == 1).astype(np.int8)  # shape: (num_ins, num_class)

        # instance * class  ins 2 label up & right
        adj_matrix[:num_ins, num_ins:] = instance_label_links

        # class * instance  label 2 ins down & left
        adj_matrix[num_ins:, :num_ins] = instance_label_links.T

        # 4. instance-instance similarity
        # epsilon = 1e-8
        print(type(Xapp))  # Xapp
        print(Xapp)  # Xapp

        # Xapp += epsilon
        sim_matrix = cosine_similarity(Xapp)  # shape: (num_ins, num_ins)

        # 5. threshold
        # instance * class pos number
        num_instance_label_links = np.sum(instance_label_links)
        total_instance_label_entries = instance_label_links.size
        target_ratio = num_instance_label_links / total_instance_label_entries * trans_ratio

        print(num_instance_label_links)
        print(target_ratio)

        total_instance_pairs = num_ins * num_ins

        # threshold ranking: (target_ratio * total_pairs) links
        flat_sim = sim_matrix.flatten()
        sorted_sim = np.sort(flat_sim)[::-1]
        cutoff_index = int(target_ratio * total_instance_pairs)
        threshold = sorted_sim[cutoff_index]

        # ins * ins similarity
        instance_adj = (sim_matrix >= threshold).astype(np.int8)

        # link self none
        np.fill_diagonal(instance_adj, 0)

        # up & left
        adj_matrix[:num_ins, :num_ins] = instance_adj

        # sparse & double
        sparse_adj = sp.csr_matrix(adj_matrix, dtype=np.float64)

        # save as .mat file with sparse
        scipy.io.savemat('../data/%s/%s_%s%din1k.mat' % (dataset, dataset, train_or_test, trans_ratio * 1000), {'net': sparse_adj})

        num_nodes = sparse_adj.shape[0]
        num_edges = sparse_adj.nnz // 2
        avg_degree = 2 * num_edges / num_nodes

        print("节点总数:", num_nodes)
        print("边数:", num_edges)
        print("平均度数:", avg_degree)
        print("稀疏邻接矩阵构建完成，形状:", sparse_adj.shape)
        print("非零元素数量:", sparse_adj.nnz)


if __name__ == '__main__':
    generate_net()
