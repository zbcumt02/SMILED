import numpy as np
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.io import savemat


def generate_net():
    dataset = 'Image'
    mat = scipy.io.loadmat('../data/%s/image.mat' % dataset)
    mat_filename = '../data/%s/finaldata.mat' % dataset

    Xapp = mat['data']  # shape: (num_ins, num_fea)
    Yapp = mat['target']  # shape: (num_class, num_ins)
    Yapp = Yapp.T

    test_ratio = 0.2
    trans_ratio = 0.001  # ratio of ins_ins change to 1 comparing with ins_label
    num_train = int(Yapp.shape[0] * (1 - test_ratio))

    X_train, X_test, Y_train, Y_test = train_test_split(Xapp, Yapp, train_size=num_train, random_state=42)

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    mat_dict = {
        'Xapp': X_train,
        'Yapp': Y_train,
        'Xgen': X_test,
        'Ygen': Y_test
    }

    savemat(mat_filename, mat_dict)

    for i in range(2):
        if i == 0:
            train_or_test = 'train'
            Xapp = X_train
            Yapp = Y_train
        else:
            train_or_test = 'test'
            Xapp = X_test
            Yapp = Y_test
        print(i)
        num_ins, num_class, num_fea = int(Yapp.shape[0]), int(Yapp.shape[1]), int(Xapp.shape[1])
        num_all = num_ins + num_class
        adj_matrix = np.zeros((num_all, num_all), dtype=np.int8)

        instance_label_links = (Yapp == 1).astype(np.int8)  # shape: (num_ins, num_class)

        adj_matrix[:num_ins, num_ins:] = instance_label_links

        adj_matrix[num_ins:, :num_ins] = instance_label_links.T

        # epsilon = 1e-8
        print(type(Xapp))
        print(Xapp)

        # Xapp += epsilon
        sim_matrix = cosine_similarity(Xapp)  # shape: (num_ins, num_ins)

        num_instance_label_links = np.sum(instance_label_links)
        total_instance_label_entries = instance_label_links.size
        target_ratio = num_instance_label_links / total_instance_label_entries * trans_ratio

        print(num_instance_label_links)
        print(target_ratio)

        total_instance_pairs = num_ins * num_ins

        flat_sim = sim_matrix.flatten()
        sorted_sim = np.sort(flat_sim)[::-1]
        cutoff_index = int(target_ratio * total_instance_pairs)
        threshold = sorted_sim[cutoff_index]

        instance_adj = (sim_matrix >= threshold).astype(np.int8)

        np.fill_diagonal(instance_adj, 0)

        adj_matrix[:num_ins, :num_ins] = instance_adj

        sparse_adj = sp.csr_matrix(adj_matrix, dtype=np.float64)

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
