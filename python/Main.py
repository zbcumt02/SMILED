import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
import warnings
import time
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
from MLL_metrics import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


warnings.simplefilter("ignore", category=RuntimeWarning)


parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data-name', default='yeast', help='dataset name')
parser.add_argument('--data-ratio', default='1in1k',
                    help='dataset similarity ratio such as 1in1k, 10in1k, 100in1k')
parser.add_argument('--max-train-num', type=int, default=10000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.2,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=2, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=15,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--num-epochs', type=int, default=100, help='epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--early-stop-patience', type=int, default=15, help='early stop patience epochs')
# Image: epoch 200 patience 30 \enron Slashdot: epoch 200 patience 50 \others: epoch 100 patience 15
# noisy label
parser.add_argument('--noisy-label', action='store_true', default=True, help='noise')
parser.add_argument('--noise-ratio', type=float, default=0.4, help='noise ratio')
parser.add_argument('--use-denoising', action='store_true', default=True,
                    help='use denoising')
parser.add_argument('--denoising-method', type=str, default='isolation_forest',
                    choices=['isolation_forest', 'lof', 'elliptic_envelope', 'dbscan'],
                    help='denoising methods')
parser.add_argument('--contamination', type=float, default=0.05,  # 0.05
                    help='Proportion of outlier pollution')
parser.add_argument('--adaptive-threshold', action='store_true', default=True,
                    help='Adaptive threshold adjustment')
# co-teaching
parser.add_argument('--use-co-teaching', action='store_true', default=True,
                    help='Use the simplified version of co-teaching for noise reduction')
parser.add_argument('--forget-rate', type=float, default=0.1,  # 0.1
                    help='Proportion of noise samples removed per round')


def main():
    args = parser.parse_args()
    args.node_name = f"{args.data_name}/finaldata"
    args.train_name = f"{args.data_name}/{args.data_name}_train{args.data_ratio}"
    args.test_name = f"{args.data_name}/{args.data_name}_test{args.data_ratio}"
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)

    '''Prepare data'''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    args.res_dir = os.path.join(args.file_dir, 'results/')

    '''Prepare node attributes'''
    node_data = sio.loadmat(os.path.join(args.file_dir, 'data/{}.mat'.format(args.node_name)))
    attributes_train = torch.tensor(node_data['Xapp'], dtype=torch.float32)
    attributes_test = torch.tensor(node_data['Xgen'], dtype=torch.float32)
    target_all = torch.tensor(node_data['Yapp'], dtype=torch.float32)
    print(target_all.shape, target_all)
    if args.noisy_label:
        noisy_target = target_all.clone()
        n_samples, n_labels = target_all.shape
        random_mask = torch.rand_like(target_all.float())
        flip_mask = random_mask < args.noise_ratio
        noisy_target = torch.where(flip_mask, -noisy_target, noisy_target)

        total_flips = flip_mask.sum().item()
        total_elements = n_samples * n_labels
        actual_flip_ratio = total_flips / total_elements

        print(f"original torch tensor: {target_all.shape}")
        print(f"noise slip ratio: {args.noise_ratio:.2%}")
        print(f"noise slip elements: {total_flips}/{total_elements} = {actual_flip_ratio:.2%}")
        print(f"original 1: {(target_all == 1).sum().item()}")
        print(f"after noise slip 1: {(noisy_target == 1).sum().item()}")
    num_class = target_all.shape[1]
    label_add = torch.zeros((num_class, attributes_train.shape[1]))
    for c in range(num_class):
        class_instances = target_all[:, c] == 1
        selected_attributes = attributes_train[class_instances]
        if selected_attributes.size(0) > 0:
            label_add[c] = selected_attributes.mean(dim=0)

    ham, ran, ap, cov, one = [], [], [], [], []

    # Build a clean sample pool outside the Loop
    clean_pools = None
    if args.use_denoising:
        print("\n" + "=" * 60)
        print("Building clean sample pools for training set...")
        print("=" * 60)

        attributes_train_np = attributes_train.numpy() if torch.is_tensor(attributes_train)\
            else attributes_train
        clean_pools = build_clean_sample_pools(
            targets_all=target_all.numpy(),
            attributes_all=attributes_train_np,
            filter_params={
                'filter_method': args.denoising_method,
                'contamination': args.contamination,
                'adaptive_threshold': args.adaptive_threshold
            }
        )
        print("Clean sample pools built successfully.")
        print("=" * 60 + "\n")

    for n in range(int(5)):
        for i in range(num_class):
            i += 0
            print(f'building classifiers: {int(i) + 1}/{num_class}')

            args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
            args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
            train_data = sio.loadmat(args.train_dir)
            test_data = sio.loadmat(args.test_dir)
            train_net = train_data['net']
            test_net = test_data['net']

            train_instance = train_net.shape[0] - num_class
            test_instance = test_net.shape[0] - num_class
            row_train = (0, train_instance)
            col_train = (train_instance + i, train_instance + i + 1)
            row_test = (0, test_instance)
            col_test = (test_instance + i, test_instance + i + 1)

            # label_add = np.zeros((num_class, attributes_train.shape[1]))
            attributes_train = np.vstack([attributes_train, label_add])
            attributes_test = np.vstack([attributes_test, label_add])
            num_feature = attributes_train.shape[1]

            if args.use_denoising and clean_pools is not None:
                print(f"\nUsing denoising module for label {i}...")

                # clean pairs
                train_pos, train_neg = sample_clean_pairs_dual(
                    label_idx=i,
                    clean_pools=clean_pools,
                    targets_train=target_all.numpy(),
                    train_net=train_net,
                    max_train_num=args.max_train_num,
                    row_range=row_train,
                    col_range=col_train
                )

                if len(train_pos[0]) < 5 or len(train_neg[0]) < 5:
                    print(f"Warning: Not enough clean samples for label {i}")
                    print("Falling back to original sampling method...")
                    train_pos, train_neg = sample_neg_mll_small_dual(
                        train_net, max_train_num=args.max_train_num,
                        row_range=row_train, col_range=col_train
                    )
            else:
                train_pos, train_neg = sample_neg_mll_small_dual(
                    train_net, max_train_num=args.max_train_num,
                    row_range=row_train, col_range=col_train
                )

            test_pos, test_neg = sample_neg_mll_small_dual(test_net, max_train_num=args.max_train_num,
                                                           row_range=row_test, col_range=col_test)
            print(f"Label {i}: Train - {len(train_pos[0])} pos, {len(train_neg[0])} neg | "
                  f"Test - {len(test_pos[0])} pos, {len(test_neg[0])} neg")

            '''Train and apply classifier'''

            a1 = train_net.copy()  # the observed network
            a2 = test_net.copy()
            a2[test_pos[0], test_pos[1]] = 0  # mask test links
            a2[test_pos[1], test_pos[0]] = 0  # mask test links
            a1 = ssp.csc_matrix(a1)
            a1.eliminate_zeros()
            a2 = ssp.csc_matrix(a2)
            a2.eliminate_zeros()

            train_graphs, max_n_label_train = links2subgraphs_dual(a1, train_pos, train_neg,
                                                                   args.hop,
                                                                   args.max_nodes_per_hop, attributes_train)
            test_graphs, max_n_label_test = links2subgraphs_dual(a2, test_pos, test_neg,
                                                                 args.hop,
                                                                 args.max_nodes_per_hop, attributes_test)
            max_n_label = max(max_n_label_train, max_n_label_test)
            train_lines = to_linegraphs(train_graphs, max_n_label)
            test_lines = to_linegraphs(test_graphs, max_n_label)
            print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))

            # Model configurations

            cmd_args.latent_dim = [32, 32, 32, 1]
            cmd_args.hidden = 128
            cmd_args.out_dim = 0
            cmd_args.dropout = True
            cmd_args.num_class = 2
            cmd_args.mode = 'gpu'
            cmd_args.num_epochs = 15
            cmd_args.learning_rate = args.lr
            cmd_args.batch_size = 50
            cmd_args.feat_dim = (max_n_label + 1 + num_feature) * 2
            cmd_args.printAUC = True
            cmd_args.attr_dim = 0

            train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
            test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)
            if args.use_co_teaching:
                print(f"\nUsing co-teaching for label {i}")

                best_preds, best_targets, best_scores = simple_co_teaching_train(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    feat_dim=cmd_args.feat_dim,
                    hidden_size=cmd_args.hidden,
                    latent_dim=cmd_args.latent_dim,
                    dropout=cmd_args.dropout,
                    lr=cmd_args.learning_rate,
                    num_epochs=args.num_epochs,
                    patience=args.early_stop_patience,
                    forget_rate=args.forget_rate,
                    device="cuda" if cmd_args.mode == 'gpu' else "cpu"
                )
            else:
                classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, cmd_args.dropout)
                if cmd_args.mode == 'gpu':
                    classifier = classifier.to("cuda")

                optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

                best_auc = 0
                best_auc_acc = 0
                best_acc = 0
                patience = args.early_stop_patience
                counter = 0

                for epoch in range(args.num_epochs + 1):
                    classifier.train()
                    avg_loss, _, _, _ = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
                    if not cmd_args.printAUC:
                        avg_loss[2] = 0.0
                    print(('\n\033[93m average training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
                    epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))

                    classifier.eval()
                    test_loss, preds, targets, scores = loop_dataset_gem(classifier, test_loader, None)
                    if not cmd_args.printAUC:
                        test_loss[2] = 0.0
                    print(('\n\033[92m average test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (
                    epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3])))

                    if best_auc_acc < test_loss[1] + test_loss[2]:
                        best_auc_acc = test_loss[1] + test_loss[2]
                        counter = 0
                        best_preds = preds
                        best_targets = targets
                        best_scores = scores
                    else:
                        counter += 1

                    if counter >= patience:
                        print(f"\n\033[91mEarly stopping triggered after {epoch} epochs\033[0m")
                        break
                    # print('best auc: %.5f best ap: %.5f' % (best_auc, best_acc))

            if i == 0:
                all_preds = torch.zeros(0, best_preds.shape[1])
                all_targets = torch.zeros(0, best_targets.shape[1])
                all_scores = torch.zeros(0, best_scores.shape[1])

            all_preds = torch.cat((all_preds, best_preds), dim=0)
            all_targets = torch.cat((all_targets, best_targets), dim=0)
            all_scores = torch.cat((all_scores, best_scores), dim=0)

            # print(all_preds.shape, all_targets.shape, all_scores.shape)
            # print(all_preds, all_targets, all_scores)

            if i > 2:
                all_preds_clone = all_preds
                all_targets_clone = all_targets
                all_scores_clone = all_scores
                scores_add = 10
                all_targets_clone[all_targets_clone == 0] = -1
                all_preds_clone[all_preds_clone == 0] = -1
                all_scores_clone = all_scores_clone.clone()
                min_values = (all_scores_clone.min(dim=0, keepdim=True)[0] +
                              all_scores_clone.min(dim=0, keepdim=True)[0] + scores_add)
                max_values = (all_scores_clone.max(dim=0, keepdim=True)[0] +
                              all_scores_clone.min(dim=0, keepdim=True)[0] + scores_add)
                all_scores_clone = (all_scores_clone + all_scores_clone.min(dim=0, keepdim=True)[0]
                                    + scores_add - min_values) / (max_values - min_values)

                hamming = Hamming_loss(all_targets_clone, all_preds_clone)
                ranking_loss = Ranking_loss(all_scores_clone, all_targets_clone)
                avg_precision = Average_precision(all_scores_clone, all_targets_clone)
                coverage = Coverage(all_scores_clone, all_targets_clone)
                one_error = One_error(all_scores_clone, all_targets_clone)

                print("Hamming Loss: %.5f\n"
                      "Ranking Loss: %.5f\n"
                      "Average Precision: %.5f\n"
                      "Coverage: %.5f\n"
                      "One Error: %.5f" % (hamming, ranking_loss, avg_precision, coverage, one_error))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = args.res_dir + f"%s_{timestamp}.pt" % args.data_name
        torch.save({
            'all_targets': all_targets,
            'all_preds': all_preds,
            'all_scores': all_scores
        }, filename)
        print("Tensors saved to results")

        scores_add = 10
        all_targets[all_targets == 0] = -1
        all_preds[all_preds == 0] = -1
        all_scores = all_scores.clone()
        min_values = all_scores.min(dim=0, keepdim=True)[0] + all_scores.min(dim=0, keepdim=True)[0] + scores_add
        max_values = all_scores.max(dim=0, keepdim=True)[0] + all_scores.min(dim=0, keepdim=True)[0] + scores_add
        all_scores = (all_scores + all_scores.min(dim=0, keepdim=True)[0] + scores_add - min_values) / (
                    max_values - min_values)

        hamming = Hamming_loss(all_targets, all_preds)
        ranking_loss = Ranking_loss(all_scores, all_targets)
        avg_precision = Average_precision(all_scores, all_targets)
        coverage = Coverage(all_scores, all_targets)
        one_error = One_error(all_scores, all_targets)

        ham.append(hamming)
        ran.append(ranking_loss)
        ap.append(avg_precision)
        cov.append(coverage)
        one.append(one_error)

        print("Total:"
              "Hamming Loss: %.5f\n"
              "Ranking Loss: %.5f\n"
              "Average Precision: %.5f\n"
              "Coverage: %.5f\n"
              "One Error: %.5f" % (hamming, ranking_loss, avg_precision, coverage, one_error))

    ham = torch.tensor(ham, dtype=torch.float32)
    ran = torch.tensor(ran, dtype=torch.float32)
    ap = torch.tensor(ap, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    one = torch.tensor(one, dtype=torch.float32)
    print("Average and Std:"
          "Hamming Loss: %.3f±%.3f\n"
          "Ranking Loss: %.3f±%.3f\n"
          "Average Precision: %.3f±%.3f\n"
          "Coverage: %.3f±%.3f\n"
          "One Error: %.3f±%.3f"
          % (mean_value(ham), std_deviation(ham), mean_value(ran), std_deviation(ran), mean_value(ap),
             std_deviation(ap), mean_value(cov), std_deviation(cov), mean_value(one), std_deviation(one)))


def mean_value(values):
    return values.mean()


def std_deviation(values):
    return values.std()


def loop_dataset_gem(classifier, loader, optimizer=None):
    total_loss = []
    all_targets = []
    all_scores = []
    preds = []

    pbar = tqdm(loader, unit='batch')

    n_samples = 0
    for batch in pbar:
        all_targets.extend(batch.y.tolist())
        logits, loss, acc, _, pred, y = classifier(batch)
        all_scores.append(logits[:, 1].cpu().detach())
        preds.append(pred.cpu().detach())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()

        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_loss.append(np.array([loss, acc]) * len(batch.y))

        n_samples += len(batch.y)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    all_targets = np.array(all_targets)
    avg_precision = average_precision_score(all_targets, all_scores)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc, avg_precision]))

    # for multi-label metrics
    all_scores = torch.tensor(all_scores).unsqueeze(0)
    all_targets = torch.tensor(all_targets).unsqueeze(0)
    all_preds = torch.cat(preds, dim=0)
    all_preds = all_preds.T

    return avg_loss, all_preds, all_targets, all_scores


def simple_co_teaching_train(train_loader, test_loader, feat_dim, hidden_size,
                             latent_dim, dropout, lr, num_epochs, patience,
                             forget_rate=0.2, device='cuda'):
    net_a = Net(feat_dim, hidden_size, latent_dim, dropout).to(device)
    net_b = Net(feat_dim, hidden_size, latent_dim, dropout).to(device)

    optimizer_a = torch.optim.Adam(net_a.parameters(), lr=lr)
    optimizer_b = torch.optim.Adam(net_b.parameters(), lr=lr)

    best_auc_acc = 0
    best_preds = None
    best_targets = None
    best_scores = None
    counter = 0

    for epoch in range(num_epochs + 1):
        net_a.train()
        train_loss_a, _, _, _ = loop_dataset_gem(net_a, train_loader, optimizer_a)
        net_b.train()
        train_loss_b, _, _, _ = loop_dataset_gem(net_b, train_loader, optimizer_b)

        net_a.eval()
        test_loss, preds, targets, scores = loop_dataset_gem(net_a, test_loader, None)

        if epoch > 0 and epoch % 2 == 0 and forget_rate > 0:
            disagreement_indices = find_disagreement_samples(net_a, net_b, train_loader, device)

            if disagreement_indices:
                train_loader = remove_noisy_samples(train_loader, disagreement_indices, forget_rate)
                print(f"Epoch {epoch}: Removed {len(disagreement_indices)} noisy samples")

        current_perf = test_loss[1] + test_loss[2]
        if current_perf >= best_auc_acc or best_preds is None:
            best_auc_acc = current_perf
            counter = 0
            best_preds = preds
            best_targets = targets
            best_scores = scores
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_preds, best_targets, best_scores


def find_disagreement_samples(net_a, net_b, data_loader, device='cuda'):
    net_a.eval()
    net_b.eval()
    disagreement_indices = []
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            logits_a, _, _, _, pred_a, _ = net_a(batch)
            logits_b, _, _, _, pred_b, _ = net_b(batch)
            for i in range(len(pred_a)):
                if pred_a[i] != pred_b[i]:
                    global_idx = batch_idx * batch_size + i
                    disagreement_indices.append(global_idx)
    return disagreement_indices


def remove_noisy_samples(original_loader, noisy_indices, forget_rate=0.2):
    """
    Remove the noise sample and create a new data loader
    """
    original_dataset = original_loader.dataset
    max_remove = int(forget_rate * len(original_loader.dataset))
    if len(noisy_indices) > max_remove:
        noisy_indices = random.sample(noisy_indices, max_remove)

    all_indices = set(range(len(original_dataset)))
    remove_set = set(noisy_indices)
    keep_indices = list(all_indices - remove_set)

    from torch.utils.data import Subset
    filtered_dataset = Subset(original_dataset, keep_indices)

    return DataLoader(filtered_dataset, batch_size=original_loader.batch_size,
                      shuffle=True, drop_last=original_loader.drop_last)


if __name__ == '__main__':
    main()
