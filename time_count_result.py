import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
import numpy as np
from utils import load_data, coarsening
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    for dataname in ['Cora']:
        for rank in [1000]:
            for v in [ 0.8, 0.9]:
                data_mol, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_optimize.npy' % (dataname, v, rank), allow_pickle=True)
                print(data_mol)
                f = open('time_count_rank_result.txt', 'a')
                f.write('%d  %.2f\n' %  (rank, v))
                for method_no in range(4, 5):
                    if method_no == 1:
                        args.coarsening_method = 'variation_edges'
                    if method_no == 2:
                        args.coarsening_method = 'algebraic_JC'
                    if method_no == 3:
                        args.coarsening_method = 'affinity_GS'
                    if method_no == 4:
                        args.coarsening_method = 'variation_edges'
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)

                    model = Net(args).to(device)
                    
                    if method_no == 4:
                        args.coarsening_method = 'Ours'
                    all_acc = []
                    for i in range(args.runs):
                        train_label = torch.zeros(data_mol.new_x.shape[0], dtype = torch.int64)
                        train_mask = torch.zeros(data_mol.new_x.shape[0]).bool()
                        val_label = torch.zeros(data_mol.new_x.shape[0], dtype = torch.int64)
                        val_mask = torch.zeros(data_mol.new_x.shape[0]).bool()
                        val_m = {}
                        train_m = {}
                        for key in m.keys():
                            if data_mol.train_mask[key] == 1:
                                train_mask[m[key]] = True
                                if m[key] not in train_m.keys():
                                    train_m[m[key]] = []
                                train_m[m[key]].append(data_mol.y[key])
                            if data_mol.val_mask[key] == 1:
                                val_mask[m[key]] = True
                            if m[key] not in val_m.keys():
                                val_m[m[key]] = []
                            val_m[m[key]].append(data_mol.y[key])
                        for key in train_m.keys():
                            # print(train_m[key])
                            train_label[key] = sorted(train_m[key], key=lambda item: (train_m[key].count(item), item))[-1]
                            # print(train_label[key])
                        for key in val_m.keys():
                            val_label[key] = sorted(val_m[key], key=lambda item: (val_m[key].count(item), item))[-1]
                        data = data_mol.to(device)
                        coarsen_features = data_mol.new_x.to(device)
                        # print(coarsen_train_labels.dtype)
                        # print(train_label.dtype)
                        coarsen_train_labels = train_label.to(device)
                        coarsen_train_mask = train_mask.to(device)
                        coarsen_val_labels = val_label.to(device)
                        coarsen_val_mask = val_mask.to(device)
                        coarsen_edge = data_mol.new_edge.to(device)
                        print((torch.unique(coarsen_train_labels[coarsen_train_mask],return_counts=True)))
                        print((torch.unique(coarsen_val_labels[coarsen_val_mask],return_counts=True)))
                            

                        if args.normalize_features:
                            coarsen_features = F.normalize(coarsen_features, p=1)
                            data.x = F.normalize(data.x, p=1)

                        model.reset_parameters()
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                        best_val_loss = float('inf')
                        val_loss_history = []
                        for epoch in range(args.epochs):

                            model.train()
                            optimizer.zero_grad()
                            out = model(coarsen_features, coarsen_edge)
                            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            pred = model(coarsen_features, coarsen_edge)
                            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

                            if val_loss < best_val_loss and epoch > args.epochs // 2:
                                best_val_loss = val_loss
                                torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

                            val_loss_history.append(val_loss)
                            if args.early_stopping > 0 and epoch > args.epochs // 2:
                                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                                if val_loss > tmp.mean().item():
                                    break

                        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
                        model.eval()
                        pred = model(data.x, data.edge_index).max(1)[1]
                        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
                        print(test_acc)
                        all_acc.append(test_acc)
                    if len(all_acc) == 0:
                        f.write('%s  ' % args.coarsening_method)
                        f.write('unable to Coarse.' + '\n')
                        continue
                    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
                    # f.write('%s  ' % args.coarsening_method)
                    f.write('ave_acc: {:.4f}'.format(np.mean(all_acc)) + ' +/- {:.4f}'.format(np.std(all_acc)) + '\n')
                f.write('\n')
                f.close()

