import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
from torch_geometric.datasets import Planetoid
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
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
            # for v in [0.3, 0.5, 0.7, 0.8, 0.9]:
            for v in [0.5, 0.7, 0.8, 0.9]:
            # for v in [0.8, 0.9]:
            # for v in [ 0.9]:
                # data_mol, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_offline.npy' % (dataname, v, rank), allow_pickle=True)
                data_mol, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_All_Simplex_1.npy' % (dataname, v, rank), allow_pickle=True)
                # data_mol, m = np.load('/home/ycmeng/ep2/Reduced_Node_Data/%s_%.2f_split%d_link_7.npy' % (dataname, v, rank), allow_pickle=True)
                print(data_mol)
                dataset = Planetoid(root='/home/ycmeng/ep1/', name=dataname)
                data = dataset[0]
                data_mol.x = data.x
                data_mol.y = data.y
                data_mol.edge_index = data.edge_index
                data_mol.train_mask = data.train_mask
                data_mol.val_mask = data.val_mask
                data_mol.test_mask = data.test_mask
                l1 = []
                l2 = []
                # print(old_to_new)
                edges = {}
                for edge_num in range(data.edge_index.size()[1]):
                    if (int(data.edge_index[0][edge_num]) in m.keys()) and (int(data.edge_index[1][edge_num]) in m.keys()):
                        node1 = m[int(data.edge_index[0][edge_num])] 
                        node2 = m[int(data.edge_index[1][edge_num])]
                        if node1 == node2:
                            continue
                        if (min(node1, node2), max(node1, node2)) in edges.keys():
                            # print("exist edge")
                            continue
                        l1.append(node1)
                        l2.append(node2)
                        l1.append(node2)  
                        l2.append(node1)
                        edges[(min(node1, node2), max(node1, node2))] = 1
                data_mol.new_edge = torch.cat((torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0)), 0)


                f = open('/home/ycmeng/ep1/time_count_rank_result.txt', 'a')
                f.write('%d  %.2f\n' %  (rank, v))
                for method_no in range(4, 5):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # device = torch.device( 'cpu')
                    args.num_features = 1433
                    args.num_classes = 7
                    # args.num_features = data_mol.x.size()[1]
                    # args.num_classes
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
                        # train_weight = torch.zeros([data_mol.new_x.shape[0], 7], dtype = torch.float).to(device)
                        # print(train_weight.shape)
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
                            # sub = sorted(train_m[key], key=lambda item: (train_m[key].count(item), item))
                            # if len(sub) > 1:
                            #     train_mask[key] = 0
                            train_label[key] = sorted(train_m[key], key=lambda item: (train_m[key].count(item), item))[-1]
                            # print(train_label[key])
                        for key in val_m.keys():
                            # sub = sorted(val_m[key], key=lambda item: (val_m[key].count(item), item))
                            # if len(sub) > 1:
                            #     val_mask[key] = 0
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
                            # print(out[coarsen_train_mask].shape)
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

