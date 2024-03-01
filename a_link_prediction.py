import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 64)
    def encode(self, data):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # 将正样本与负样本拼接 shape:[2,272]
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()# 将模型和数据送入设备
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels# 训练函数，每次训练重新采样负边，计算模型损失，反向传播误差，更新模型参数
def train(model, data):
    model.train()
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,num_neg_samples=data.train_pos_edge_index.size(1),  # 负采样数量根据正样本
                                       force_undirected=True,)  # 得到负采样shape: [2,136]
    neg_edge_index = neg_edge_index.to(device)
    optimizer.zero_grad()
    z = model.encode(data)  # 利用正样本训练学习得到每个节点的特征 shape:[34, 64]
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # [272] 利用正样本和负样本 按位相乘 求和  (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)  # [272] 前136个是1，后136个是0
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)  # binary_cross_entropy_with_logits会自动计算link_logits的sigmoid
    loss.backward()
    optimizer.step()
    return loss# 测试函数，评估模型在验证集和测试集上的预测准确率
@torch.no_grad()
def test(model, data):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode(data)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs# 训练模型，每次训练完，输出模型在验证集和测试集上的预测准确率
    

import copy
from utils1 import load_data, coarsening
import warnings
 
# 禁用所有警告信息的显示
warnings.filterwarnings("ignore")
file_name = 'a_link_result_re.txt'
for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
# for v in [0.9]:
    # for dataname in ['dblp']:
    for dataname in ['Cora', 'Citeseer', 'dblp', 'pubmed']:
        for rank in [1000]:
            data_sub, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_optimize.npy' % (dataname, 0.1, rank), allow_pickle=True)
            num_nodes = data_sub.x.size()[0]
        # for v in [0.8]:
            for method in ['ours', 'variation_neighborhoods', 'algebraic_JC', 'affinity_GS', 'variation_edges']:
            # for method in ['variation_edges']:
                if v == 0:
                    if method != 'ours':
                        continue
                try:
                    if method == 'ours':
                        data, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_link.npy' % (dataname, v, rank), allow_pickle=True)
                        # data.num_features = 3
                        # data.edge_attr = None# 构造节点特征矩阵（原网络不存在节点特征）
                        # data.x = torch.ones((data.num_nodes, data.num_features), dtype=torch.float32)
                        # 分割训练边集、验证边集（默认占比0.05）以及测试边集（默认占比0.1）
                        data_ori = copy.deepcopy(data)
                        data_ori = train_test_split_edges(data_ori)
                        data.x = data.new_x
                        data.edge_index = data.new_edge
                        data = train_test_split_edges(data)
                        # print(data)
                        # 构造一个简单的图卷积神经网络（两层），包含编码（节点嵌入）、解码（分数预测）等操作
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model, data, data_ori = Net().to(device), data.to(device), data_ori.to(device)
                        # 指定优化器
                        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)# 将训练集中的正边标签设置为1，负边标签设置为0
                        best_val_perf = test_perf = 0
                        for epoch in range(1, 100):
                            train_loss = train(model, data)
                            val_perf, tmp_test_perf = test(model, data)
                            if val_perf > best_val_perf:
                                best_val_perf = val_perf
                                test_perf = tmp_test_perf
                                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                                # print(epoch, train_loss, best_val_perf, test_perf)
                        val_perf, test_perf = test(model, data_ori)
                        log = 'Dataname: {:s}, v: {:.1f}, Method: {:s}, Val: {:.4f}, Test: {:.4f}\n'
                        f = open(file_name, 'a')
                        f.write(log.format(dataname, v, method, best_val_perf, test_perf))
                        f.close()
                        print(log.format(dataname, v, method, best_val_perf, test_perf))# 利用训练好的模型计算网络中剩余所有边的分数
                        # z = model.encode(data)
                        # final_edge_index = model.decode_all(z)
                    else:
                        if dataname == 'Cora':
                            dataset = 'cora'
                        elif dataname == 'Citeseer':
                            dataset = 'citeseer'
                        else:
                            dataset = dataname
                        num_try = 0
                        now_size = 100000
                        flag = 0
                        current = v
                        while now_size > num_nodes*(1-v):
                            num_features, num_classes, candidate, C_list, Gc_list = coarsening(dataset, current, method)
                            data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
                                dataset, candidate, C_list, Gc_list, 'fixed')
                            num_try += 1
                            now_size = coarsen_features.size()[0]
                            current=current + 0.01
                            print(now_size)
                            if num_try==20:
                                flag = 1
                                break
                        # if flag == 1:
                        #     continue
                        data.x = coarsen_features
                        data.edge_index = coarsen_edge
                        # print(data)
                        data_ori, m = np.load('/home/ycmeng/ep1/Reduced_Node_Data/%s_%.2f_split%d_optimize.npy' % (dataname, v, rank), allow_pickle=True)
                        # data.num_features = 3
                        # data.edge_attr = None# 构造节点特征矩阵（原网络不存在节点特征）
                        # data.x = torch.ones((data.num_nodes, data.num_features), dtype=torch.float32)
                        # 分割训练边集、验证边集（默认占比0.05）以及测试边集（默认占比0.1）
                        data_ori = train_test_split_edges(data_ori)
                        data = train_test_split_edges(data)
                        # print(data)
                        # 构造一个简单的图卷积神经网络（两层），包含编码（节点嵌入）、解码（分数预测）等操作
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model, data, data_ori = Net().to(device), data.to(device), data_ori.to(device)
                        # 指定优化器
                        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)# 将训练集中的正边标签设置为1，负边标签设置为0
                        best_val_perf = test_perf = 0
                        for epoch in range(1, 100):
                            train_loss = train(model, data)
                            val_perf, tmp_test_perf = test(model, data)
                            if val_perf > best_val_perf:
                                best_val_perf = val_perf
                                test_perf = tmp_test_perf
                                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                                # print(epoch, train_loss, best_val_perf, test_perf)
                        val_perf, test_perf = test(model, data_ori)
                        log = 'Dataname: {:s}, v: {:.1f}, Method: {:s}, Val: {:.4f}, Test: {:.4f}\n'
                        f = open(file_name, 'a')
                        f.write(log.format(dataname, v, method, best_val_perf, test_perf))
                        f.close()
                        print(log.format(dataname, v, method, best_val_perf, test_perf))# 利用训练好的模型计算网络中剩余所有边的分数
                        # z = model.encode(data)
                        # final_edge_index = model.decode_all(z)
                except:
                    log = 'Something wrong with Dataname: {:s}, v: {:.1f}, Method: {:s}\n'
                    f = open(file_name, 'a')
                    f.write(log.format(dataname, v, method))
                    f.close()
                    print(log.format(dataname, v, method))