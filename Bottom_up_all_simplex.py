from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
import logging
import scipy.sparse as sp
# from graph_coarsening.coarsening_utils import *
import networkx as nx
# from find_maximal_clique import find_max_cliques, find_max_cliques_nx
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
import random
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
import copy
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue, Queue
import pdb
from sklearn.metrics import roc_auc_score
import warnings
import pandas as pd
import shutil, os
import os.path as osp
import torch
import sys
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
import psutil

def find_max_cliques_nx(graph, giveup_size): 
    # OtN = {}
    # NtO = {}
    # for i, node in enumerate(list(graph)):
    #     OtN[node] = i+1
    #     NtO[i+1] = node
    # new_graph = nx.relabel_nodes(graph, OtN)
    # num_node = len(new_graph)
    # C = [0 for _ in range(num_node + 1)]
    # C, res = pivoter(new_graph, C)
    # for i in range(num_node + 1):
    #     if (C[i]):
    #         print(i, C[i])
    res = []
    cnt_clique = 0
    for clique in nx.find_cliques(graph):
        if len(clique) > 1:
            res.append(list(clique))
            cnt_clique += 1
            if cnt_clique > giveup_size:
                break
    for i in range(len(res)):
        res[i].sort()
    # for i in range(len(res)):
    #     for j in range(len(res[i])):
    #         res[i][j] = NtO[res[i][j]]
    #     res[i].sort()
    # print(res)
    return res


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def extract_components(H):
    if H.A.shape[0] != H.A.shape[1]:
        H.logger.error('Inconsistent shape to extract components. '
                       'Square matrix required.')
        return None

    if H.is_directed():
        raise NotImplementedError('Directed graphs not supported yet.')

    graphs = []

    visited = np.zeros(H.A.shape[0], dtype=bool)

    while not visited.all():
        stack = set([np.nonzero(~visited)[0][0]])
        comp = []

        while len(stack):
            v = stack.pop()
            if not visited[v]:
                comp.append(v)
                visited[v] = True

                stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                  if not visited[idx]]))

        comp = sorted(comp)
        G = H.subgraph(comp)
        G.info = {'orig_idx': comp}
        graphs.append(G)

    return graphs


maximun_memory = 0
warnings.filterwarnings('ignore')
sys.setrecursionlimit(100000)


class Evaluator:
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0,
                                keep_default_na=False)
        if not self.name in meta_info:
            print(self.name)
            error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(meta_info.keys())
            raise ValueError(error_mssg)

        self.num_tasks = int(meta_info[self.name]['num tasks'])
        self.eval_metric = meta_info[self.name]['eval metric']

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return {'acc': sum(acc_list) / len(acc_list)}


class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  ## original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0,
                                 keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
                not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygNodePropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                        additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

        else:
            data = \
                read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                               additional_node_files=additional_node_files,
                               additional_edge_files=additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip',
                                         header=None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class nodes:
    def __init__(self, index):  # 初始化 包含左端点所在的x, y坐标以及长度l
        self.index = index
        self.vanished = 0  # 这个点上重叠的点数
        self.edgenode = 0  # 这个点上的边数
        self.nodes = []  # 这个点上重叠的点的列表
        self.ed_van = []  # 这个点上由于消去边获得的特征
        self.train_node = False  # 是否为训练节点
        self.remain = False  # 是否需要保留
        self.recast = 0  # 是否为训练节点
        self.label = -1

    def __lt__(self, other):  # 为了堆优化重载运算符"<"
        if self.edgenode < other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast < other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished < other.vanished:
                return True
            return False

    def __gt__(self, other):  # 为了堆优化重载运算符">"
        if self.edgenode > other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast > other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished > other.vanished:
                return True
            return False

    def __eq__(self, other):  # 为了堆优化重载运算符"="
        if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished == other.vanished:
            return True
        else:
            return False


class simplex:
    def __init__(self, index, rank):  # 初始化 包含左端点所在的x, y坐标以及长度l
        self.index = index
        self.rank = rank
        self.nodes = []
        self.high = set()
        self.low = set()
        self.maximum_face = set()
        self.face = set()
        self.num_maximum_face = 0
        self.vanished = False


# 检查是否为极大团
def check_maximal(simplexs, node_list, maximal_cliques):
    base_keys = list(simplexs[0][node_list[0]].maximum_face)
    flag = True
    for key in base_keys:
        maximal_clique = maximal_cliques[key].nodes
        if set(node_list).issubset(set(maximal_clique)):
            flag = False
            # print("                checked", node_list, maximal_clique)
            break
    return flag


# def check_maximal(graph, node_list):
#     ll = list(graph.neighbors(node_list[0]))
#     for i in range(1, len(node_list)):
#         ll = list(np.intersect1d(ll, list(graph.neighbors(node_list[i]))))
#     return (len(ll) == 0)


# 构建其对应的子图
def get_subgraphs(formal_list, need_num_node, remain_list, subgraph_list):
    if need_num_node == 0:
        subgraph_list.append(formal_list)
        return subgraph_list
    for i in range(len(remain_list) - need_num_node + 1):
        subgraph_list = get_subgraphs(formal_list + [remain_list[i]], need_num_node - 1, remain_list[i + 1:],
                                      subgraph_list)
    return subgraph_list


def clear_self_and_higher(rank, idx_r, simplexs):
    simplexs[rank][idx_r].vanished = True
    snapshot = list(simplexs[rank][idx_r].high)
    for idx in snapshot:
        simplexs = clear_self_and_higher(rank + 1, idx, simplexs)
        # 将这些边对应的另外的点进行一下关系上的更新
        # (这些边必然也只属于一个即将collapse的 maximal clique)
        # (之后会对maximal clique对应的face直接进行更新)
    for other_node in simplexs[rank][idx_r].low:
        simplexs[rank - 1][other_node].high.discard(idx_r)
    simplexs[rank][idx_r].low = set()
    return simplexs


# 对一个点和其对应的 maximal clique 进行collapse
def node_collapse(graph, node, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
                  max_cluster_size, recoil_percent, back_edges, simplex_list):
    # 首先把这个点从图上删除
    # print("node_collapse", node, list(graph.neighbors(node)), simplexs[0][nodes_to_index[0][node]].vanished, maximal_cliques[list(simplexs[0][nodes_to_index[0][node]].maximum_face)[0]].nodes)
    # print("node_collapse", node)
    # print(node, list(graph.neighbors(node)), simplexs[0][nodes_to_index[0][node]].vanished, simplexs[0][nodes_to_index[0][node]].num_maximum_face, [maximal_cliques[cqc].nodes for cqc in list(simplexs[0][nodes_to_index[0][node]].maximum_face)])
    # if 2160 in nodes_to_index[0].keys():
    #     print(2160, list(simplexs[0][nodes_to_index[0][2160]].maximum_face))
    # print(graph.nodes())
    # print(graph.edges())
    global n
    # if 146118 in list(graph.nodes()):
    #     print('node_collapse')
    #     print("node", node)
    #     print("mal_cliques", [c for c in nx.find_cliques(graph) if node in c])
    # print(n[node].train_node)
    # if len(list(graph.nodes())) <= max_cluster_size * recoil_percent:
    #     return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list
    # print("vanishing", node, list(graph.neighbors(node))[0] )
    # n_neighbor = list(graph.neighbors(node))
    # reduced_node_list.append((node, n_neighbor[random.randint(0, len(n_neighbor)-1)]))
    # print(n[node].train_node, n[add_target].train_node)
    nei_node = list(graph.neighbors(node))
    add_target = nei_node[0]
    flag_train = 1
    if n[node].train_node:
        flag_train = 0
    #     # for nei in list(graph.neighbors(node)):
    #     #     if n[nei].train_node == 0:
    #     #         flag_train = 1
    #     #         add_target = nei
    #     #         break
    # else:
        for nei in nei_node:
            if n[nei].train_node == 0:
                flag_train = 1
                add_target = nei
    if not flag_train:
        reduced_node_list.append((node, -2))
        for nei in list(graph.neighbors(node)):
            reduced_edge_list.append((nei, -1))
            back_edges.append((node, nei))
        # print(graph.nodes())
        graph.remove_node(node)
        idx_0 = nodes_to_index[0][node]
        simplexs = clear_self_and_higher(0, idx_0, simplexs)
        # simplexs[0][idx_0].vanished = True
        # for idx in simplexs[0][idx_0].high:
        #     simplexs[1][idx].vanished = True
        #     # 将这些边对应的另外的点进行一下关系上的更新
        #     # (这些边必然也只属于一个即将collapse的 maximal clique)
        #     # (之后会对maximal clique对应的face直接进行更新)
        #     for other_node in simplexs[1][idx].low:
        #         if other_node == idx_0:
        #             continue
        #         simplexs[0][other_node].high.discard(idx)
        q1 = Queue()
        # q2 = Queue()
        set_node = set()
        set_edge = set()
        # 对即将进行collapse的maximal
        # print(list(simplexs[0][idx_0].maximum_face))
        idx_max = list(simplexs[0][idx_0].maximum_face)[0]
        maximal_clique = maximal_cliques[idx_max].nodes
        # print("vanished", maximal_clique)
        # print(node, idx_0, maximal_clique)
        pos = maximal_clique.index(idx_0)
        new_maximal_clique = maximal_clique[:pos] + maximal_clique[pos + 1:]

        # 如果去掉node后, 其不再是一个maximal clique
        maximal_cliques[idx_max].vanished = True
        faces = maximal_cliques[idx_max].face
        for pair in faces:
            simplexs[pair[0]][pair[1]].maximum_face.discard(idx_max)
            simplexs[pair[0]][pair[1]].num_maximum_face -= 1
        numnode = len(new_maximal_clique)
        if numnode > 1:
            if (check_maximal(simplexs, new_maximal_clique, maximal_cliques)):
                # print("remain", maximal_clique)
                # 如果去掉node后, 其仍为一个maximal clique
                # 直接将这个maximal clique相关的信息更新掉
                ll = new_maximal_clique
                key = tuple(ll)
                maximal_cliques[idx_max] = simplex(idx_max, numnode - 1)
                maximal_cliques[idx_max].nodes = ll
                nodes_to_index["max"][key] = idx_max
                for len1 in range(1, numnode):
                    if (len1 - 1) not in simplexs.keys():
                        continue
                    add_clique = get_subgraphs([], len1, ll, [])
                    for face1 in add_clique:
                        if len(face1) > 1:
                            kk = tuple(face1)
                        else:
                            kk = face1[0]
                        maximal_cliques[idx_max].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                        simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(idx_max)
                        simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1

        for pair in faces:
            if pair[0] == 0 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                q1.put(simplexs[pair[0]][pair[1]].nodes[0])
            if pair[0] == 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                q2.put(pair[1])
            if pair[0] > 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                simplex_list.append(pair)
        graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = collapse_other_node(
            q1, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
            max_cluster_size, recoil_percent, back_edges, simplex_list)

        return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list

    same_label_candidate = []
    for nei in nei_node:
        if n[nei].train_node == 0 and n[nei].label == n[node].label:
            same_label_candidate.append(nei)
    if len(same_label_candidate) > 0:
        add_target = same_label_candidate[0]

    # for nei in nei_node:
    #     if n[nei].train_node == 0:
    #         same_label_candidate.append(nei)
    # add_target = random.choice(same_label_candidate)

    reduced_node_list.append((node, add_target))
    n[add_target].train_node = n[add_target].train_node or n[node].train_node
    for nei in list(graph.neighbors(node)):
        reduced_edge_list.append((nei, -1))
    # print(graph.nodes())
    graph.remove_node(node)

    # 对该点所对应的所有边进行清理
    idx_0 = nodes_to_index[0][node]

    simplexs = clear_self_and_higher(0, idx_0, simplexs)
    # simplexs[0][idx_0].vanished = True
    # for idx in simplexs[0][idx_0].high:
    #     simplexs[1][idx].vanished = True
    #     # 将这些边对应的另外的点进行一下关系上的更新
    #     # (这些边必然也只属于一个即将collapse的 maximal clique)
    #     # (之后会对maximal clique对应的face直接进行更新)
    #     for other_node in simplexs[1][idx].low:
    #         simplexs[0][other_node].high.discard(idx)

    q1 = Queue()
    # q2 = Queue()
    set_node = set()
    set_edge = set()
    # 对即将进行collapse的maximal
    # print(list(simplexs[0][idx_0].maximum_face))
    idx_max = list(simplexs[0][idx_0].maximum_face)[0]
    maximal_clique = maximal_cliques[idx_max].nodes
    # print(node, idx_0, maximal_clique)
    pos = maximal_clique.index(idx_0)
    # print("vanished", maximal_clique)
    new_maximal_clique = maximal_clique[:pos] + maximal_clique[pos + 1:]

    # 如果去掉node后, 其不再是一个maximal clique
    maximal_cliques[idx_max].vanished = True
    faces = maximal_cliques[idx_max].face

    for pair in faces:
        simplexs[pair[0]][pair[1]].maximum_face.discard(idx_max)
        simplexs[pair[0]][pair[1]].num_maximum_face -= 1
        # if pair[0] == 0:
        #     if len([c for c in nx.find_cliques(graph) if pair[1] in c]) != simplexs[0][pair[1]].num_maximum_face:
        #         print("1!!!")
        # if pair[0] == 1:
        #     if len([c for c in nx.find_cliques(graph) if (simplexs[pair[0]][pair[1]].nodes[0] in c and simplexs[pair[0]][pair[1]].nodes[1] in c)]) != simplexs[pair[0]][pair[1]].num_maximum_face:
        #         print("1!!!")
        # if len(simplexs[pair[0]][pair[1]].maximum_face) != simplexs[pair[0]][pair[1]].num_maximum_face:
        #     print("1!!!")

    numnode = len(new_maximal_clique)
    if numnode > 1:
        if (check_maximal(simplexs, new_maximal_clique, maximal_cliques)):
            # print("remain", maximal_clique)
            # 如果去掉node后, 其仍为一个maximal clique
            # 直接将这个maximal clique相关的信息更新掉
            ll = new_maximal_clique
            key = tuple(ll)
            maximal_cliques[idx_max] = simplex(idx_max, numnode - 1)
            maximal_cliques[idx_max].nodes = ll
            nodes_to_index["max"][key] = idx_max
            for len1 in range(1, numnode):
                if (len1 - 1) not in simplexs.keys():
                    continue
                add_clique = get_subgraphs([], len1, ll, [])
                for face1 in add_clique:
                    if len(face1) > 1:
                        kk = tuple(face1)
                    else:
                        kk = face1[0]
                    maximal_cliques[idx_max].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                    simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(idx_max)
                    simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1

    # print("mal_cliques", [c for c in nx.find_cliques(graph) if node in c])
    for pair in faces:
        if pair[0] == 0 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            q1.put(simplexs[pair[0]][pair[1]].nodes[0])
        if pair[0] == 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            q2.put(pair[1])
        if pair[0] > 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            simplex_list.append(pair)

    # for node in set_node:
    #     q1.put(node)
    #
    # for edge in set_edge:
    #     q2.put(edge)
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = collapse_other_node(
        q1, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
        max_cluster_size, recoil_percent, back_edges, simplex_list)
    # graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list = collapse_other_edge(q2, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list


# 对一个边和其对应的 maximal clique 进行collapse, 或者是对一个边直接进行删除
def edge_collapse(graph, nodes, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
                  max_cluster_size, recoil_percent, back_edges, simplex_list):
    # 首先把这条边从图上删除
    # print(nodes)
    # print(nx.is_frozen(graph))
    # print(simplexs[0][1862])
    # print(graph.nodes())
    # print(graph.edges())
    # if len(list(graph.nodes())) <= max_cluster_size * recoil_percent:
    #     return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list
    q1 = Queue()
    q2 = Queue()
    graph.remove_edge(nodes[0], nodes[1])
    reduced_edge_list.append((nodes[0], nodes[1]))
    # print(nodes[0], nodes[1])
    # 将这个边对应的点进行关系更新
    idx_1 = nodes_to_index[1][tuple(nodes)]
    simplexs = clear_self_and_higher(1, idx_1, simplexs)
    # simplexs[1][idx_1].vanished = True
    # for other_node in simplexs[1][idx_1].low:
    #     simplexs[0][other_node].high.discard(idx_1)

    # 如果这条边是一个maximal
    if tuple(nodes) in nodes_to_index["max"].keys():
        idx_max = nodes_to_index["max"][tuple(nodes)]
        maximal_cliques[idx_max].vanished = True
        for pair in maximal_cliques[idx_max].face:
            simplexs[pair[0]][pair[1]].num_maximum_face -= 1
            if pair[0] == 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                q1.put(simplexs[pair[0]][pair[1]].nodes[0])
            simplexs[pair[0]][pair[1]].maximum_face.discard(idx_max)
            # if len(simplexs[pair[0]][pair[1]].maximum_face) != simplexs[pair[0]][pair[1]].num_maximum_face:
            #     print("4!!!")
            # if pair[0] == 0:
            #     if len([c for c in nx.find_cliques(graph) if pair[1] in c]) != simplexs[0][pair[1]].num_maximum_face:
            #         print("4!!!")
            # if pair[0] == 1:
            #     if len([c for c in nx.find_cliques(graph) if (simplexs[pair[0]][pair[1]].nodes[0] in c and simplexs[pair[0]][pair[1]].nodes[1] in c)]) != simplexs[pair[0]][pair[1]].num_maximum_face:
            #         print("4!!!")

    idx_00 = nodes_to_index[0][nodes[0]]
    idx_01 = nodes_to_index[0][nodes[1]]

    # print(simplexs[0][1862])
    # additional_maximum_clique = []
    # 如果edge_collapse是由于直接删除边导致的, 那么可能会出现其存在多个 maximum face 的情况
    snapshot = list(simplexs[1][idx_1].maximum_face)
    # if 146118 in list(graph.nodes()):
    #     print('maximal faces', snapshot)
    #     for i in range(len(snapshot)):
    #         print('nodes of mamximal', maximal_cliques[snapshot[i]].nodes)
    all_pair = []
    for i in range(len(snapshot)):
        idx_max = snapshot[i]
        maximal_clique = maximal_cliques[idx_max].nodes
        faces = maximal_cliques[idx_max].face
        for pair in faces:
            simplexs[pair[0]][pair[1]].maximum_face.discard(idx_max)
            simplexs[pair[0]][pair[1]].num_maximum_face -= 1
    for i in range(len(snapshot)):
        idx_max = snapshot[i]
        maximal_clique = maximal_cliques[idx_max].nodes
        faces = maximal_cliques[idx_max].face
        # print(maximal_clique)
        # 如果该maximal clique本身就已经是一条边了, 那就不需要再存入maximal clique了
        if len(maximal_clique) == 2:
            continue
        pos = maximal_clique.index(idx_00)
        # print(idx_00)
        maximal_clique1 = maximal_clique[:pos] + maximal_clique[pos + 1:]
        pos = maximal_clique.index(idx_01)
        # print(idx_01)
        maximal_clique2 = maximal_clique[:pos] + maximal_clique[pos + 1:]
        cliques = []
        if (check_maximal(simplexs, maximal_clique1, maximal_cliques)):
            cliques.append(maximal_clique1)
        if (check_maximal(simplexs, maximal_clique2, maximal_cliques)):
            cliques.append(maximal_clique2)
        # print("            maximal_split", maximal_clique, cliques)
        # print(cliques)
        # print(simplexs[0][1862])
            # if len(simplexs[pair[0]][pair[1]].maximum_face) != simplexs[pair[0]][pair[1]].num_maximum_face:
            #     print("5!!!")
            # if pair[0] == 0:
            #     if len([c for c in nx.find_cliques(graph) if pair[1] in c]) != simplexs[0][pair[1]].num_maximum_face:
            #         print("5!!!")
            # if pair[0] == 1:
            #     if len([c for c in nx.find_cliques(graph) if (simplexs[pair[0]][pair[1]].nodes[0] in c and simplexs[pair[0]][pair[1]].nodes[1] in c)]) != simplexs[pair[0]][pair[1]].num_maximum_face:
            #         print("5!!!")
        # 如果产生了新的maximum clique, 我们就把他加入进去
        if len(cliques) > 0:
            ll = cliques[0]
            numnode = len(ll)
            key = tuple(ll)
            maximal_cliques[idx_max] = simplex(idx_max, numnode - 1)
            maximal_cliques[idx_max].nodes = ll
            nodes_to_index["max"][key] = idx_max
            for len1 in range(1, numnode):
                if (len1 - 1) not in simplexs.keys():
                    continue
                add_clique = get_subgraphs([], len1, ll, [])
                for face1 in add_clique:
                    if len(face1) > 1:
                        kk = tuple(face1)
                    else:
                        kk = face1[0]
                    # if len1 >= 4:
                    #     print(nodes_to_index[len1 - 1].keys())
                    maximal_cliques[idx_max].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                    simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(idx_max)
                    simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1

            if len(cliques) > 1:
                for new_maximal_clique in cliques[1:]:
                    ll = new_maximal_clique
                    numnode = len(ll)
                    key = tuple(ll)
                    maximal_cliques[cnt["max"]] = simplex(cnt["max"], numnode - 1)
                    maximal_cliques[cnt["max"]].nodes = ll
                    nodes_to_index["max"][key] = cnt["max"]
                    for len1 in range(1, numnode):
                        if (len1 - 1) not in simplexs.keys():
                            continue
                        add_clique = get_subgraphs([], len1, ll, [])
                        for face1 in add_clique:
                            if len(face1) > 1:
                                kk = tuple(face1)
                            else:
                                kk = face1[0]
                            maximal_cliques[cnt["max"]].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                            simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(cnt["max"])
                            simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1
                    cnt["max"] += 1
        # # 如果所有的新的clique都不是maximum clique
        # if cliques == []:
        # 更新其所有的face对应的maximum face的数量
        if len(cliques) == 0:
            maximal_cliques[idx_max].vanished = True
        for pair in faces:
            all_pair.append(pair)
            if pair[0] == 0 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                q1.put(simplexs[pair[0]][pair[1]].nodes[0])
            if pair[0] == 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                q2.put(pair[1])
            if pair[0] > 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
                simplex_list.append(pair)

    # if 146118 in list(graph.nodes()):
    #     print("faces:")
    #     for pair in all_pair:
    #         if pair[0] == 0:
    #             print("nodes", simplexs[pair[0]][pair[1]].nodes)
    #             # print(graph.nodes())
    #             print("find_maximal", [c for c in nx.find_cliques(graph) if pair[1] in c])
    #             print("num_maximal", simplexs[pair[0]][pair[1]].num_maximum_face)
    #         if pair[0] == 1:
    #             print("nodes", simplexs[pair[0]][pair[1]].nodes)
    #             print("find_maximal", [c for c in nx.find_cliques(graph) if (simplexs[pair[0]][pair[1]].nodes[0] in c and simplexs[pair[0]][pair[1]].nodes[1] in c)])
    #             print("num_maximal", simplexs[pair[0]][pair[1]].num_maximum_face)
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = collapse_other_node(
        q1, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
        max_cluster_size, recoil_percent, back_edges, simplex_list)
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list = collapse_other_edge(
        q2, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
        max_cluster_size, recoil_percent, back_edges, simplex_list)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list


# 对一个simplex和其对应的 maximal clique 进行collapse
def simplex_collapse(graph, rank, nodes, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list,
                     reduced_edge_list,
                     max_cluster_size, recoil_percent, back_edges):
    # print("simplex_collapse")
    # 首先把这条边从图上删除
    # print(nx.is_frozen(graph))
    # print(simplexs[0][1862])
    # print(graph.nodes())
    # print(graph.edges())
    # if len(list(graph.nodes())) <= max_cluster_size * recoil_percent:
    #     return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges
    q1 = Queue()
    q2 = Queue()
    simplex_list = []

    # print(nodes[0], nodes[1])
    # 将这个边对应的点进行关系更新
    idx_r = nodes_to_index[rank][tuple(nodes)]
    # simplexs[rank][idx_r].vanished = True

    simplexs = clear_self_and_higher(rank, idx_r, simplexs)

    # print(simplexs[0][1862])
    # additional_maximum_clique = []
    # 如果edge_collapse是由于直接删除边导致的, 那么可能会出现其存在多个 maximum face 的情况
    idx_max = list(simplexs[rank][idx_r].maximum_face)[0]
    maximal_clique = maximal_cliques[idx_max].nodes
    faces = maximal_cliques[idx_max].face
    # print(maximal_clique)
    # 如果该maximal clique本身就已经是一条边了, 那就不需要再存入maximal clique了
    cliques = []
    for node1 in simplexs[rank][idx_r].nodes:
        pos = maximal_clique.index(node1)
        # print(idx_00)
        new_maximal_clique = maximal_clique[:pos] + maximal_clique[pos + 1:]
        if (check_maximal(simplexs, new_maximal_clique, maximal_cliques)):
            cliques.append(new_maximal_clique)

    # print(cliques)
    for pair in faces:
        simplexs[pair[0]][pair[1]].maximum_face.discard(idx_max)
        simplexs[pair[0]][pair[1]].num_maximum_face -= 1
    # 如果产生了新的maximum clique, 我们就把他加入进去

    if len(cliques) > 0:
        ll = cliques[0]
        numnode = len(ll)
        key = tuple(ll)
        maximal_cliques[idx_max] = simplex(idx_max, numnode - 1)
        maximal_cliques[idx_max].nodes = ll
        nodes_to_index["max"][key] = idx_max
        for len1 in range(1, numnode):
            if (len1 - 1) not in simplexs.keys():
                continue
            add_clique = get_subgraphs([], len1, ll, [])
            for face1 in add_clique:
                if len(face1) > 1:
                    kk = tuple(face1)
                else:
                    kk = face1[0]
                maximal_cliques[idx_max].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(idx_max)
                simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1

        if len(cliques) > 1:
            for new_maximal_clique in cliques[1:]:
                ll = new_maximal_clique
                numnode = len(ll)
                key = tuple(ll)
                maximal_cliques[cnt["max"]] = simplex(cnt["max"], numnode - 1)
                maximal_cliques[cnt["max"]].nodes = ll
                nodes_to_index["max"][key] = cnt["max"]
                for len1 in range(1, numnode):
                    if (len1 - 1) not in simplexs.keys():
                        continue
                    add_clique = get_subgraphs([], len1, ll, [])
                    for face1 in add_clique:
                        if len(face1) > 1:
                            kk = tuple(face1)
                        else:
                            kk = face1[0]
                        maximal_cliques[cnt["max"]].face.add((len1 - 1, nodes_to_index[len1 - 1][kk]))
                        simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].maximum_face.add(cnt["max"])
                        simplexs[len1 - 1][nodes_to_index[len1 - 1][kk]].num_maximum_face += 1
                cnt["max"] += 1
    # # 如果所有的新的clique都不是maximum clique
    # if cliques == []:
    # 更新其所有的face对应的maximum face的数量
    if len(cliques) == 0:
        maximal_cliques[idx_max].vanished = True
    for pair in faces:
        if pair[0] == 0 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            q1.put(simplexs[pair[0]][pair[1]].nodes[0])
        if pair[0] == 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            q2.put(pair[1])
        if pair[0] > 1 and simplexs[pair[0]][pair[1]].num_maximum_face == 1:
            simplex_list.append(pair)

    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = collapse_other_node(
        q1, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
        max_cluster_size, recoil_percent, back_edges, simplex_list)
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list = collapse_other_edge(
        q2, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
        max_cluster_size, recoil_percent, back_edges, simplex_list)
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges = collapse_other_simplex(
        simplex_list, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
        max_cluster_size, recoil_percent, back_edges)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges


# 构造所有的极大团以及0,1-simplex
def Get_all_cliques(graph, nodes_to_index):
    # print("Get_all_cliques")
    # print(graph.nodes())
    # print(graph.edges())
    print(graph)
    simplexs = {}
    cnt = {}
    maximal_cliques = {}
    print("Finding.....")
    cliques = find_max_cliques_nx(graph, 100000)
    cnt["max"] = 0
    nodes_to_index["max"] = {}
    print("最大团数量", len(cliques))
    clique_count = len(cliques)
    # for ll in cliques:
    #     # print(ll)
    #     numnode = len(ll)
    #     if numnode > 1:
    #         clique_count += 1
    if clique_count >= 100000:
        cnt["max"] = clique_count
        return simplexs, maximal_cliques, nodes_to_index, cnt
    if 0 not in simplexs.keys():
        # 构造 0 阶单纯形
        simplexs[0] = {}
        nodes_to_index[0] = {}
        for node in graph.nodes():
            simplexs[0][node] = simplex(node, 0)
            simplexs[0][node].nodes.append(node)
            nodes_to_index[0][node] = node
    if 1 not in simplexs.keys():
        # 构造 1 阶单纯形
        cnt[1] = 0
        simplexs[1] = {}
        nodes_to_index[1] = {}
        for edge in graph.edges():
            simplexs[1][cnt[1]] = simplex(cnt[1], 1)
            simplexs[0][edge[0]].high.add(cnt[1])
            simplexs[0][edge[1]].high.add(cnt[1])
            simplexs[1][cnt[1]].low.add(edge[0])
            simplexs[1][cnt[1]].low.add(edge[1])
            simplexs[1][cnt[1]].nodes = [min(edge[0], edge[1]), max(edge[0], edge[1])]
            nodes_to_index[1][(min(edge[0], edge[1]), max(edge[0], edge[1]))] = cnt[1]
            cnt[1] += 1
    for ll in cliques:
        # print(ll)
        numnode = len(ll)
        maximum_size = 1000
        # 过大的团就分裂成小团（应该不会出现这种情况）
        if numnode > maximum_size:
            new_clique = get_subgraphs([], maximum_size, ll, [])
            for lll in new_clique:
                # print(ll)
                numnode = len(lll)
                key = tuple(lll)
                if numnode > 1:
                    maximal_cliques[cnt["max"]] = simplex(cnt["max"], numnode - 1)
                    maximal_cliques[cnt["max"]].nodes = lll
                    nodes_to_index["max"][key] = cnt["max"]
                    for i in range(numnode):
                        # print(ll[i])
                        maximal_cliques[cnt["max"]].face.add((0, nodes_to_index[0][lll[i]]))
                        simplexs[0][lll[i]].maximum_face.add(cnt["max"])
                        simplexs[0][lll[i]].num_maximum_face += 1
                        # if len([c for c in nx.find_cliques(graph) if lll[i] in c]) != simplexs[0][lll[i]].num_maximum_face:
                        #     print("10!!!")
                        if numnode > 2:
                            for j in range(i + 1, numnode):
                                # print((ll[i], ll[j]))
                                maximal_cliques[cnt["max"]].face.add((1, nodes_to_index[1][(lll[i], lll[j])]))
                                simplexs[1][nodes_to_index[1][(lll[i], lll[j])]].maximum_face.add(cnt["max"])
                                simplexs[1][nodes_to_index[1][(lll[i], lll[j])]].num_maximum_face += 1
                                # if len([c for c in nx.find_cliques(graph) if ((lll[i] in c) and (lll[j] in c))]) != simplexs[1][nodes_to_index[1][(lll[i], lll[j])]].num_maximum_face:
                                #     print("11!!!")
                    cnt["max"] += 1
            continue
        key = tuple(ll)
        if numnode > 1:
            maximal_cliques[cnt["max"]] = simplex(cnt["max"], numnode - 1)
            maximal_cliques[cnt["max"]].nodes = ll
            nodes_to_index["max"][key] = cnt["max"]
            for i in range(numnode):
                # print(ll[i])
                maximal_cliques[cnt["max"]].face.add((0, nodes_to_index[0][ll[i]]))
                simplexs[0][ll[i]].maximum_face.add(cnt["max"])
                simplexs[0][ll[i]].num_maximum_face += 1
                # if len([c for c in nx.find_cliques(graph) if ll[i] in c]) != simplexs[0][ll[i]].num_maximum_face:
                #     print("12!!!")
                if numnode > 2:
                    for j in range(i + 1, numnode):
                        # print((ll[i], ll[j]))
                        maximal_cliques[cnt["max"]].face.add((1, nodes_to_index[1][(ll[i], ll[j])]))
                        simplexs[1][nodes_to_index[1][(ll[i], ll[j])]].maximum_face.add(cnt["max"])
                        simplexs[1][nodes_to_index[1][(ll[i], ll[j])]].num_maximum_face += 1
                        # if len([c for c in nx.find_cliques(graph) if ((ll[i] in c) and (ll[j] in c))]) != simplexs[1][nodes_to_index[1][(ll[i], ll[j])]].num_maximum_face:
                        #     print("13!!!")
            cnt["max"] += 1
    # simplex[0] 为点集合,  simplex[1] 为边集合, maximal_cliques 为极大团集合,  cnt 为当前的极大团数量
    return simplexs, maximal_cliques, nodes_to_index, cnt


# 对simplex进行查看, 如果满足collapse的条件就将其collapse
def collapse_other_simplex(simplex_list, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list,
                           reduced_edge_list,
                           max_cluster_size, recoil_percent, back_edges):
    # print("collapse_other_simplex")
    # print(graph.nodes())
    # print(graph.edges())
    for r, key in simplex_list:
        # print("simplex", key, r)
        if simplexs[r][key].vanished == False and simplexs[r][key].num_maximum_face == 1:
            graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges = simplex_collapse(
                graph, r, simplexs[r][key].nodes, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list,
                reduced_edge_list,
                max_cluster_size, recoil_percent, back_edges)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges


# 对队列中的边进行查看, 如果满足collapse的条件就将其collapse
def collapse_other_edge(q, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
                        max_cluster_size, recoil_percent, back_edges, simplex_list):
    # print("collapse_other_edge")
    # print(graph.nodes())
    # print(graph.edges())
    while not q.empty():
        key = q.get()
        if simplexs[1][key].vanished == False and simplexs[1][key].num_maximum_face == 1:
            # print(nx.is_frozen(graph))
            graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list = edge_collapse(
                graph, simplexs[1][key].nodes, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list,
                reduced_edge_list, max_cluster_size, recoil_percent, back_edges, simplex_list)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list


# TODO: 突然疑惑一个问题，这个队列里面的simplex有没有可能在过程中指向了不同的simplex？
# 对队列中的点进行查看, 如果满足collapse的条件就将其collapse
def collapse_other_node(q, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
                        q2, max_cluster_size, recoil_percent, back_edges, simplex_list):
    # print("collapse_other_edge")
    # print(graph.nodes())
    # print(graph.edges())
    while not q.empty():
        key = q.get()
        if simplexs[0][nodes_to_index[0][key]].vanished == False and simplexs[0][
            nodes_to_index[0][key]].num_maximum_face == 1:
            # print(nx.is_frozen(graph))
            # print(3)
            graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = node_collapse(
                graph, key, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
                max_cluster_size, recoil_percent, back_edges, simplex_list)
    return graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list


# 对图进行预处理, 能collapse的边全部collapse
def Graph_reduce(graph: nx.Graph, rank, simplexs, nodes_to_index, max_cluster_size, recoil_percent, back_edges):
    # print("Graph_reduce")
    # print(graph.nodes())
    # print(graph.edges())
    simplexs, maximal_cliques, nodes_to_index, cnt = Get_all_cliques(graph, nodes_to_index)

    if cnt["max"] >= 100000:
        # 将邻接矩阵的index和原图上的节点名映射
        reduced_node_list = []
        reduced_edge_list = []
        # print("Var. Nei Reducing")
        # m_keep = {}
        # for new, ori in enumerate(list(graph.nodes())):
        #     # print(new, ori)
        #     m_keep[new] = ori
        # # print(nx.adjacency_matrix(graph))

        # adj_matrix = nx.adjacency_matrix(graph)
        # adj_array = adj_matrix.toarray()

        # G = gsp.graphs.Graph(W=adj_array)
        # components = extract_components(G)
        # # print('the number of subgraphs is', len(components))
        # candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
        # number = 0
        # C_list=[]
        # Gc_list=[]
        # while number < len(candidate):
        #     H = candidate[number]
        #     if len(H.info['orig_idx']) > 10:
        #         C, Gc, Call, Gall = coarsen(H, r=0.5, method="variation_neighborhoods")
        #         C_list.append(C)
        #         Gc_list.append(Gc)
        #     number += 1
        # # 将分割后的子图的index和邻接矩阵的index映射
        # o_to_n = {}
        # print("cnt[2] >= 10000")
        # print([C.shape for C in C_list])
        # for number in range(len(candidate)):
        #     get_target = {}
        #     H = candidate[number]
        #     keep = H.info['orig_idx']
        #     # print(keep)
        #     if len(H.info['orig_idx']) > 10:
        #         C = C_list[number]
        #         Gc = Gc_list[number]
        #         # print(C.shape)
        #         # print(type(C))
        #         rows, cols = C.nonzero()
        #         # print(len(rows))
        #         for i, j in zip(rows, cols):
        #             # print(i, j, C[i, j])s
        #             if m_keep[keep[j]] in o_to_n.keys():
        #                 print("map ", m_keep[keep[j]], "to", i, "origin is", o_to_n[m_keep[keep[j]]])
        #                 print("???")
        #             if i not in get_target.keys():
        #                 get_target[i] = m_keep[keep[j]]
        #             o_to_n[m_keep[keep[j]]] = get_target[i]
        #             # if o_to_n[m_keep[keep[j]]]
        # for key in o_to_n.keys():
        #     if key != o_to_n[key]:
        #         for nei in graph.neighbors(key):
        #             if o_to_n[key] != nei:
        #                 graph.add_edge(o_to_n[key], nei)
        #         graph.remove_node(key)
        #         reduced_node_list.append((key, o_to_n[key]))
        flag_none = 0

        print(cnt)
        return graph, flag_none, simplexs, maximal_cliques, nodes_to_index, reduced_node_list, reduced_edge_list, cnt, back_edges

    reduced_node_list = []
    reduced_edge_list = []

    flag_none = 0
    # print(simplexs[0])
    q1 = Queue()
    q2 = Queue()
    simplex_list = []
    for key in simplexs[0].keys():
        if simplexs[0][key].num_maximum_face == 1:
            q1.put(simplexs[0][key].nodes[0])
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = collapse_other_node(
        q1, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, q2,
        max_cluster_size, recoil_percent, back_edges, simplex_list)

    if len(list(graph.nodes())) <= max_cluster_size * recoil_percent:
        print(cnt)
        return graph, flag_none, simplexs, maximal_cliques, nodes_to_index, reduced_node_list, reduced_edge_list, cnt, back_edges

    for key in simplexs[1].keys():
        if simplexs[1][key].num_maximum_face == 1:
            q2.put(key)

    # print(nx.is_frozen(graph))
    graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list, back_edges, simplex_list = collapse_other_edge(
        q2, graph, simplexs, nodes_to_index, maximal_cliques, cnt, reduced_node_list, reduced_edge_list,
        max_cluster_size, recoil_percent, back_edges, simplex_list)

    if len(list(graph.nodes())) <= max_cluster_size * recoil_percent:
        print(cnt)
        return graph, flag_none, simplexs, maximal_cliques, nodes_to_index, reduced_node_list, reduced_edge_list, cnt, back_edges

    for r in range(2, rank + 1):
        # 构造 r 阶单纯形
        cliques = []
        cnt[r] = 0
        simplexs[r] = {}
        nodes_to_index[r] = {}
        all_r_clique = set()
        ready_to_collapse = set()
        for max_key in maximal_cliques.keys():
            if maximal_cliques[max_key].vanished:
                continue
            max_clique = maximal_cliques[max_key].nodes
            if len(max_clique) > r:
                cliques = get_subgraphs([], r + 1, max_clique, [])

            for r_clique in cliques:
                # TODO: 验证一下这里的 r_clique是不是排好序的
                if tuple(r_clique) not in nodes_to_index[r].keys():
                    simplexs[r][cnt[r]] = simplex(cnt[r], r)
                    simplexs[r][cnt[r]].nodes = r_clique
                    nodes_to_index[r][tuple(r_clique)] = cnt[r]
                    cnt[r] += 1
                if len(max_clique) > r + 1:
                    simplexs[r][nodes_to_index[r][tuple(r_clique)]].maximum_face.add(max_key)
                    simplexs[r][nodes_to_index[r][tuple(r_clique)]].num_maximum_face += 1
            if cnt[r] >= 100000 and r > 2:
                del simplexs[r]
                del nodes_to_index[r]
                break

        if cnt[r] == 0 or (cnt[r] >= 100000 and r > 2):
            break
        for r_key in simplexs[r].keys():
            r_clique = simplexs[r][r_key].nodes
            for max_key in simplexs[r][r_key].maximum_face:
                maximal_cliques[max_key].face.add((r, r_key))
            if simplexs[r][r_key].num_maximum_face == 1:
                ready_to_collapse.add((r, r_key))
            facets = get_subgraphs([], r, r_clique, [])
            for facet in facets:
                simplexs[r - 1][nodes_to_index[r - 1][tuple(facet)]].high.add(r_key)
                simplexs[r][r_key].low.add(nodes_to_index[r - 1][tuple(facet)])

    print(cnt)
    return graph, flag_none, simplexs, maximal_cliques, nodes_to_index, reduced_node_list, reduced_edge_list, cnt, back_edges


def write_graph(data, num_cluster, graphs, n, old_graph, node_to_cluster, back_edges, v, max_cluster_size):
    res = copy.deepcopy(data).to(device)
    print("time:", time.time_ns())
    start = time.time_ns()
    res.new_x = copy.deepcopy(res.x)
    flag = 0
    old_to_new = {}
    new_to_old = {}
    # cnt_node = 0
    rem_flag = 0
    for cluster in range(0, num_cluster):
        # print(graph[cluster])
        # print(graphs[cluster].nodes())
        for node in graphs[cluster].nodes():
            # print(node)
            # cnt_node += 1
            if flag == 0:  # 记录当前点数量
                # res.new_x = copy.deepcopy(res.x[node].unsqueeze(0))
                old_to_new[node] = flag
                new_to_old[flag] = node
                for vanished in n[node].nodes:
                    old_to_new[vanished] = old_to_new[node]
                flag += 1
            else:
                # res.new_x = torch.cat((res.new_x, res.x[node].unsqueeze(0)), 0)
                old_to_new[node] = flag
                new_to_old[flag] = node
                for vanished in n[node].nodes:
                    old_to_new[vanished] = old_to_new[node]
                flag += 1
    add_feature = []
    end = time.time_ns()
    # print((end - start) // 1000000)
    l1 = []
    l2 = []
    edges = {}
    for edge in back_edges:
        if int(edge[0]) not in old_to_new.keys():
            old_to_new[int(edge[0])] = flag
            new_to_old[flag] = int(edge[0])
            add_feature.append((int(edge[0]), flag))
            for vanished in n[int(edge[0])].nodes:
                old_to_new[vanished] = old_to_new[int(edge[0])]
            flag += 1
        if int(edge[1]) not in old_to_new.keys():
            old_to_new[int(edge[1])] = flag
            new_to_old[flag] = int(edge[1])
            add_feature.append((int(edge[1]), flag))
            for vanished in n[int(edge[1])].nodes:
                old_to_new[vanished] = old_to_new[int(edge[1])]
            flag += 1
        node1 = old_to_new[int(edge[0])]
        node2 = old_to_new[int(edge[1])]
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
    # 将图之间的边加上
    for old_graph_edge in old_graph.edges():
        node1 = int(old_graph_edge[0])
        node2 = int(old_graph_edge[1])
        if node_to_cluster[node1] == node_to_cluster[node2]:
            continue
        if node1 not in old_to_new.keys():
            print("!!!", node1)
            old_to_new[node1] = flag
            new_to_old[flag] = node1
            add_feature.append((node1, flag))
            flag += 1
        if node2 not in old_to_new.keys():
            print("!!!", node2)
            old_to_new[node2] = flag
            new_to_old[flag] = node2
            add_feature.append((node2, flag))
            flag += 1
        node1 = old_to_new[node1]
        node2 = old_to_new[node2]
        if (min(node1, node2), max(node1, node2)) in edges.keys():
            # print("exist edge")
            continue
        l1.append(node1)
        l2.append(node2)
        l1.append(node2)
        l2.append(node1)
        edges[(min(node1, node2), max(node1, node2))] = 1
    # print(cnt)

    for cluster in range(0, num_cluster):
        for edge in graphs[cluster].edges():
            l1.append(old_to_new[edge[0]])
            l2.append(old_to_new[edge[1]])
            l1.append(old_to_new[edge[1]])
            l2.append(old_to_new[edge[0]])
    res.new_edge = torch.cat((torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0)), 0)
    # print(old_to_new)
    global dataname
    end = time.time_ns()
    res.new_x = torch.zeros([flag, data.x.size()[1]])
    # print(flag)
    # 对于图上的每一个点
    end = time.time_ns()
    # print((end - start) // 1000000)
    for cluster in range(0, num_cluster):
        for node in graphs[cluster].nodes():
            # sub 为当前保留下的节点的特征
            sub = res.x[node]
            # 将 vanished 中的所有的节点特征相加, 同时将其指向当前的点
            for vanished in n[node].nodes:
                old_to_new[vanished] = old_to_new[node]
                sub = sub + res.x[vanished]
            # # 将由于去边所获得的特征加入
            # for vanished in n[node].ed_van:
            #     sub = sub + res.x[vanished]
            # 取其中均值
            # if len(n[node].nodes) != 0 or len(n[node].ed_van) != 0:
            res.new_x[old_to_new[node]] = sub / (float((1 + len(n[node].nodes) + len(n[node].ed_van))))
            # res.new_x[old_to_new[node]] = sub / (float((1+len(n[node].nodes)+len(n[node].ed_van))) ** 0.5)
    # print((end - start) // 1000000)
    for pair in add_feature:
        sub = res.x[pair[0]]
        for vanished in n[pair[0]].nodes:
            old_to_new[vanished] = old_to_new[pair[0]]
            sub = sub + res.x[vanished]
        res.new_x[old_to_new[pair[0]]] = sub / (float((1 + len(n[pair[0]].nodes) + len(n[pair[0]].ed_van))))
    res.x = 0
    res.y = 0
    res.edge_index = 0
    res.train_mask = 0
    res.val_mask = 0
    res.test_mask = 0
    np.save('./Reduced_Node_Data/%s_%.2f_split%d_All_Simplex_1.npy' % (dataname, v, max_cluster_size), (res.cpu(), old_to_new))
    print(v, res)
    return res, old_to_new, new_to_old


def Modify(data, percentage, device, max_rank, reduce_edge, n, graphs, flag_none, simplexs,
           nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank,
           num_cluster, finished, node_to_cluster, max_cluster_size, high_sim, low_sim, old_graph, maximal_cliques,
           cnts, recoil_percent, ratio_list, back_edges):
    target_n = len(data.x) * percentage
    start = time.time_ns()
    rest_list = []
    for i in range(0, num_cluster):
        if finished[i] == 0:
            rest_list.append(i)
    while n_vanished < target_n:
        # print(graphs[0].nodes())
        # print(graphs[0].edges())
        # print('edge_vanishing_cnt', edge_vanishing_cnt)
        # for i in simplexs[0].keys():
        #     print(i, len(simplexs[0][i].keys()))
        #     for idx in simplexs[0][i].keys():
        #         print(simplexs[0][i][idx].nodes, simplexs[0][i][idx].vanished)
        # for key in n:0
        #     print('node', key.index, key.edgenode)

        # 重新将图进行分块
        if sum(finished) == num_cluster:
            # print("!!!!")
            # print('before')
            # for cluster in range(0, num_cluster):
            #     print(graphs[cluster])
            # time.sleep(1)
            # 生成一个 old_to_new, 记录节点的 collapse 关系
            old_to_new = {}
            for cluster in range(0, num_cluster):
                for node in graphs[cluster].nodes():
                    old_to_new[node] = node
                    for vanished in n[node].nodes:
                        old_to_new[vanished] = node

            # 新建一张图, 将之前的子图中剩余的点和边全部加入进去
            graph = nx.Graph()
            for g in graphs:
                graph.add_nodes_from(g.nodes)
                graph.add_edges_from(g.edges)

            # print(graph.edges())
            # print(node_to_cluster)

            # 将之前抛弃的边补充回去
            for edge in back_edges:
                if int(edge[0]) not in old_to_new.keys():
                    old_to_new[int(edge[0])] = int(edge[0])
                    for vanished in n[int(edge[0])].nodes:
                        old_to_new[vanished] = int(edge[0])
                if int(edge[1]) not in old_to_new.keys():
                    old_to_new[int(edge[1])] = int(edge[1])
                    for vanished in n[int(edge[1])].nodes:
                        old_to_new[vanished] = int(edge[1])
                node1 = old_to_new[int(edge[0])]
                node2 = old_to_new[int(edge[1])]
                # 如果构成自环了那么就忽略它
                if node1 == node2:
                    continue
                # 并不需要检查, 因为实际上 networkx 无向图一条边重复添加无影响
                '''
                if (min(node1, node2), max(node1, node2)) in edges.keys():
                    # print("exist edge")
                    continue
                '''
                graph.add_edge(node1, node2)

            # 将之前的图中的所有的子图之间的边加入进去
            for old_graph_edge in old_graph.edges():
                node1 = int(old_graph_edge[0])
                node2 = int(old_graph_edge[1])
                if node_to_cluster[node1] == node_to_cluster[node2]:
                    continue
                if node1 not in old_to_new.keys():
                    print("!!!")
                    old_to_new[node1] = node1
                    for vanished in n[node1].nodes:
                        old_to_new[vanished] = node1
                if node2 not in old_to_new.keys():
                    print("!!!")
                    old_to_new[node2] = node2
                    for vanished in n[node2].nodes:
                        old_to_new[vanished] = node2
                graph.add_edge(old_to_new[node1], old_to_new[node2])

            #
            random_reduce = 0
            back_edges = []
            # print(graph.nodes())

            # 如果说里面只剩下了训练节点, 那么就随机对图中各节点进行 collapse
            for edge in graph.edges():
                if (not n[edge[0]].train_node) or (not n[edge[1]].train_node):
                    random_reduce = 1
            if random_reduce == 0:
                while n_vanished < target_n:
                    for i in range(num_cluster):
                        target = n[random.choice(list(graphs[i].nodes()))]  # 改为图中的任意一个点
                        target_neighbor = random.choice(list(graphs[i].nodes()))
                        # print(type(target.index), type(target_neighbor))
                        id = target.index
                        while target_neighbor == id or (n[target.index].train_node and n[target_neighbor].train_node):
                            target_neighbor = random.choice(list(graphs[i].nodes()))

                        n[target_neighbor].nodes = n[target_neighbor].nodes + target.nodes + [target.index]
                        n[target_neighbor].ed_van = n[target_neighbor].ed_van + target.ed_van
                        n[target_neighbor].vanished += target.vanished
                        n[target_neighbor].train_node = (n[target.index].train_node or n[target_neighbor].train_node)
                        n[target_neighbor].remain = (n[target.index].remain or n[target_neighbor].remain)
                        graphs[i].remove_node(target.index)
                        # print(neighbor[0], target.index)
                        n_vanished += 1
                        if n_vanished >= target_n:
                            res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph,
                                                                      node_to_cluster, back_edges, percentage,
                                                                      max_cluster_size)
                            pos_r = ratio_list.index(percentage)
                            if pos_r != len(ratio_list) - 1:
                                percentage = ratio_list[pos_r + 1]
                                target_n = len(data.x) * percentage
                            else:
                                return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges

                        node_vanished[target.index] = 1
                continue
            print("!!!", graph)
            # print(graph.edges())

            com_nodes = []
            components = nx.connected_components(graph)
            for component in components:
                com = list(component)
                # print("components", len(com), com)
                if len(com) >= 10:
                    com_nodes = com_nodes + com
                else:
                    add_flag = 0
                    for node in com:
                        if n[node].remain:
                            add_flag = 1
                            break
                    if add_flag == 1:
                        com_nodes = com_nodes + com
                    else:
                        for node in com:
                            n[node].vanished = True
                        n_vanished += len(com)
            graph = nx.Graph(graph.subgraph(com_nodes))
            old_graph = nx.Graph(graph)
            # if n_vanished >= target_n:
            #     res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph, node_to_cluster, back_edges, percentage, max_cluster_size)
            #     pos_r = ratio_list.index(percentage)
            #     if pos_r != len(ratio_list) - 1:
            #         percentage = ratio_list[pos_r + 1]
            #         target_n = len(data.x) * percentage
            #     else:
            #         return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph

            # 如果这张图里面还存在能够collapse的可能
            num_nodes = data.x.size()[0]
            new_node_to_cluster = list(0 for i in range(0, num_nodes))
            new_graphs = []
            cnt_cluster_node = 0
            current_nodes = []
            i = 1
            # print(max_cluster_size, cnt_cluster_node)

            # 重新对得到的图进行子图切分
            q1 = Queue()
            for node in graph.nodes():
                # print(node, not new_node_to_cluster[node], len(list(graph[node].keys())))
                if (not new_node_to_cluster[node]) and len(list(graph[node].keys())) != 0:
                    q1.put(copy.deepcopy(node))
                    while not q1.empty():
                        node = q1.get()
                        # print('here is the node', node)
                        if new_node_to_cluster[node] != 0:
                            continue
                            # print('here is the node', node)
                        current_nodes.append(node)
                        new_node_to_cluster[node] = i
                        # print('update', node)
                        for nei in n[node].nodes:
                            # print('update', nei)
                            new_node_to_cluster[nei] = i
                        # print(current_nodes)
                        cnt_cluster_node += 1
                        if max_cluster_size <= cnt_cluster_node:
                            # while not q1.empty():
                            #     node = q1.get()
                            #     if new_node_to_cluster[node] != 0:
                            #         continue
                            #     if len(list(graph[node].keys())) == 0:
                            #         current_nodes.append(node)
                            #         new_node_to_cluster[node] = i
                            #         # print(current_nodes)
                            #         cnt_cluster_node += 1
                            #         graph.remove_node(node)
                            q1 = Queue()
                            i += 1
                            new_graphs.append(nx.Graph(graph.subgraph(current_nodes)))
                            cluster_node.append(current_nodes)
                            current_nodes = []
                            cnt_cluster_node = 0
                            break
                        neighbor = list(graph[node].keys())
                        for node2 in neighbor:
                            if not new_node_to_cluster[node2]:
                                q1.put(copy.deepcopy(node2))
            # print(current_nodes)
            # print(new_node_to_cluster)
            for node in graph.nodes():
                if not new_node_to_cluster[node]:
                    current_nodes.append(node)
                    new_node_to_cluster[node] = i
                    # print('node', node)
                    # print(n[node].nodes)
                    for nei in n[node].nodes:
                        new_node_to_cluster[nei] = i
                    # print(current_nodes)
                    cnt_cluster_node += 1
                    if max_cluster_size <= cnt_cluster_node:
                        new_graphs.append(nx.Graph(graph.subgraph(current_nodes)))
                        cluster_node.append(current_nodes)
                        i += 1
                        cnt_cluster_node = 0
                        current_nodes = []
            if len(current_nodes) != 0:
                new_graphs.append(nx.Graph(graph.subgraph(current_nodes)))
                cluster_node.append(current_nodes)

            # num_cluster = len(graphs)
            global maximun_memory
            pid = os.getpid()

            process = psutil.Process(pid)
            memory_info = process.memory_info()
            memory = memory_info.rss
            if memory - current_memory > maximun_memory:
                maximun_memory = memory - current_memory
                # print('3 Memory cost:', memory-current_memory)

            # 对新的这些子图进行初始化
            graphs = new_graphs
            # print('len(new_graphs)', len(new_graphs))
            node_to_cluster = new_node_to_cluster
            # print('node_to_cluster', new_node_to_cluster)
            num_cluster = len(graphs)
            finished = list(0 for i in range(0, num_cluster))
            # print(num_cluster)
            nodes_to_index = list({} for i in range(0, num_cluster))
            reduced_rank = list(1 for i in range(0, num_cluster))
            simplexs = list({} for i in range(0, num_cluster))
            flag_none = list(0 for i in range(0, num_cluster))
            maximal_cliques = list({} for i in range(0, num_cluster))
            # print(num_cluster, len(graphs), len(simplexs), len(nodes_to_index))
            if num_cluster > 1:
                for i in range(num_cluster):
                    reduced_node_list = []
                    reduced_edge_list = []
                    # print(i)
                    # print(graphs[i])
                    # print(simplexs[i])
                    # print(nodes_to_index[i])
                    graphs[i], flag_none[i], simplexs[i], maximal_cliques[i], nodes_to_index[
                        i], reduced_node_list, reduced_edge_list, cnts[i], back_edges = Graph_reduce(graphs[i], 6,
                                                                                                     simplexs[i],
                                                                                                     nodes_to_index[i],
                                                                                                     max_cluster_size,
                                                                                                     recoil_percent,
                                                                                                     back_edges)


                    for pair in reduced_node_list:
                        if pair[1] == -2:
                            n[pair[0]].recast = 0
                        else:
                            n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                            # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                            n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                            if n[pair[1]].label != n[pair[0]].label:
                                n[pair[1]].label = -1
                            n[pair[1]].recast = 0
                            # n[n[pair[0]].index].edgenode -= 1
                            # n[pair[1]].edgenode -= 1
                            n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                            n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                            n[pair[1]].vanished += n[pair[0]].vanished
                            n_vanished += 1
                            node_vanished[n[pair[0]].index] = 1
                    if len(list(graphs[i].nodes())) <= max_cluster_size * recoil_percent and len(graphs) > 1:
                        finished[i] = 1
                        simplexs[i] = {}
                        maximal_cliques[i] = {}


                    if n_vanished >= target_n:
                        res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph,
                                                                  node_to_cluster, back_edges, percentage,
                                                                  max_cluster_size)
                        pos_r = ratio_list.index(percentage)
                        if pos_r != len(ratio_list) - 1:
                            percentage = ratio_list[pos_r + 1]
                            target_n = len(data.x) * percentage
                        else:
                            return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges


            else:
                for i in range(num_cluster):
                    reduced_node_list = []
                    reduced_edge_list = []
                    graphs[i], flag_none[i], simplexs[i], maximal_cliques[i], nodes_to_index[
                        i], reduced_node_list, reduced_edge_list, cnts[i], back_edges = Graph_reduce(graphs[i], 6,
                                                                                                     simplexs[i],
                                                                                                     nodes_to_index[i],
                                                                                                     max_cluster_size,
                                                                                                     0, back_edges)
                    # print("In Modify 2", reduced_node_list)
                    for pair in reduced_node_list:
                        if pair[1] == -2:
                            n[pair[0]].recast = 0
                        else:
                            n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                            # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                            n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                            n[pair[1]].recast = 0
                            if n[pair[1]].label != n[pair[0]].label:
                                n[pair[1]].label = -1
                            # n[n[pair[0]].index].edgenode -= 1
                            # n[pair[1]].edgenode -= 1
                            n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                            n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                            n[pair[1]].vanished += n[pair[0]].vanished
                            n_vanished += 1
                            node_vanished[n[pair[0]].index] = 1
                    if n_vanished >= target_n:
                        res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph,
                                                                  node_to_cluster, back_edges, percentage,
                                                                  max_cluster_size)
                        pos_r = ratio_list.index(percentage)
                        if pos_r != len(ratio_list) - 1:
                            percentage = ratio_list[pos_r + 1]
                            target_n = len(data.x) * percentage
                        else:
                            return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges

            for i in range(num_cluster):
                # print(graphs[i])
                # print(graphs[i].edges())
                for node in graphs[i].nodes():
                    n[node].edgenode = 0
                    n[node].recast = 0
                for edge in graphs[i].edges():
                    n[edge[0]].edgenode += 1
                    n[edge[1]].edgenode += 1

            q = list(PriorityQueue() for i in range(0, num_cluster))
            for i in range(num_cluster):
                for node in graphs[i].nodes():
                    if n[node].edgenode != 0:
                        q[i].put(copy.deepcopy(n[node]))
            # print('after')
            # print(len(graphs))
            # for cluster in range(0, num_cluster):
            #     print(graphs[cluster])
            rest_list = []
            for i in range(0, num_cluster):
                if finished[i] == 0:
                    rest_list.append(i)

        # 如果collapse的目标依然没有达到
        for cluster in rest_list:
            # print('here')
            # print(n[2].index, n[2].train_node)
            if n_vanished % 500 == 0:
                print('n_vanished', n_vanished)
            if finished[cluster] == 1:
                pos = rest_list.index(cluster)
                rest_list = rest_list[:pos] + rest_list[pos + 1:]
                continue
            if n_vanished >= target_n:
                break
            graph = nx.Graph(graphs[cluster])
            if len(list(graph.nodes())) <= max_cluster_size * recoil_percent and len(graphs) > 1:
                finished[cluster] = 1
                simplexs[cluster] = {}
                maximal_cliques[cluster] = {}
                pos = rest_list.index(cluster)
                rest_list = rest_list[:pos] + rest_list[pos + 1:]

                continue
            if cnts[cluster]["max"] >= 100000:
                print("too many cliques!", cnts[cluster]["max"], len(rest_list))
                # print("Var. Nei Reducing")
                # m_keep = {}
                # for new, ori in enumerate(list(graph.nodes())):
                #     # print(new, ori)
                #     m_keep[new] = ori
                # # print(nx.adjacency_matrix(graph))

                # adj_matrix = nx.adjacency_matrix(graph)
                # adj_array = adj_matrix.toarray()

                # G = gsp.graphs.Graph(W=adj_array)
                # components = extract_components(G)
                # # print('the number of subgraphs is', len(components))
                # candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
                # number = 0
                # C_list=[]
                # Gc_list=[]
                # ratio = 1 - (float(max_cluster_size* recoil_percent) / len(list(graph.nodes())))
                # while number < len(candidate):
                #     H = candidate[number]
                #     if len(H.info['orig_idx']) > 10:
                #         C, Gc, Call, Gall = coarsen(H, r=ratio, method="variation_neighborhoods")
                #         C_list.append(C)
                #         Gc_list.append(Gc)
                #     number += 1

                # o_to_n = {}
                # reduced_node_list = []
                # reduced_edge_list = []
                # print([C.shape for C in C_list])
                # for number in range(len(candidate)):
                #     get_target = {}
                #     H = candidate[number]
                #     keep = H.info['orig_idx']
                #     # print(keep)
                #     if len(H.info['orig_idx']) > 10:
                #         C = C_list[number]
                #         Gc = Gc_list[number]
                #         # print(C.shape)
                #         # print(type(C))
                #         rows, cols = C.nonzero()
                #         # print(len(rows))
                #         for i, j in zip(rows, cols):
                #             # print(i, j, C[i, j])s
                #             if m_keep[keep[j]] in o_to_n.keys():
                #                 print("map ", m_keep[keep[j]], "to", i, "origin is", o_to_n[m_keep[keep[j]]])
                #                 print("???")
                #             if i not in get_target.keys():
                #                 get_target[i] = m_keep[keep[j]]
                #             o_to_n[m_keep[keep[j]]] = get_target[i]
                #             # if o_to_n[m_keep[keep[j]]]
                # for key in o_to_n.keys():
                #     if key != o_to_n[key]:
                #         for nei in graph.neighbors(key):
                #             if o_to_n[key] != nei:
                #                 graph.add_edge(o_to_n[key], nei)
                #         graph.remove_node(key)
                #         reduced_node_list.append((key, o_to_n[key]))
                # flag_none = 0
                # for pair in reduced_node_list:
                #     n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                #     # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                #     n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                #     n[pair[1]].recast = 0
                #     n[n[pair[0]].index].edgenode -= 1
                #     n[pair[1]].edgenode -= 1
                #     n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                #     n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                #     n[pair[1]].vanished += n[pair[0]].vanished
                #     n_vanished += 1
                #     node_vanished[n[pair[0]].index] = 1
                finished[cluster] = 1
                # for node in graph.nodes():
                #     n[node].edgenode = 0
                #     n[node].recast = 0
                # for edge in graph.edges():
                #     n[edge[0]].edgenode += 1
                #     n[edge[1]].edgenode += 1

                pos = rest_list.index(cluster)
                rest_list = rest_list[:pos] + rest_list[pos + 1:]
                graphs[cluster] = graph
                simplexs[cluster] = {}
                maximal_cliques[cluster] = {}
                continue
            if n_vanished % 1000 == 0:
                try:
                    file = open('Reduce_new.txt', mode='a', encoding='utf-8')
                except:
                    print(f'open analyse file error!')
                file.write('n_vanished: %lf' % n_vanished)
                file.write('\n')
                file.close()
            if q[cluster].empty():
                # print("all %d nodes are vanished" % n_vanished)
                finished[cluster] = 1
                simplexs[cluster] = {}
                maximal_cliques[cluster] = {}
                pos = rest_list.index(cluster)
                rest_list = rest_list[:pos] + rest_list[pos + 1:]
                continue
            # q[cluster].join()
            target = q[cluster].get()
            # print(target.index, target.vanished, target.edgenode, target.nodes, target.ed_van, target.train_node)
            # print(target.index)
            # print('target node:', target.index, target.edgenode)
            # others = []
            # while not q[cluster].empty():
            #     other = q[cluster].get()
            #     print('other node:', other.index, other.edgenode, target<other, other<target)
            #     others.append(other)
            # q[cluster].put(target)
            # for other in others:
            #     q[cluster].put(other)
            # target = q[cluster].get()
            # print('target node:', target.index, target.edgenode)
            # print(target)
            if target.index in node_vanished.keys():
                # print("??")
                continue
            if len(target.nodes) != len(n[target.index].nodes) or target.vanished != n[
                target.index].vanished or target.edgenode != n[target.index].edgenode or target.recast != n[
                target.index].recast:
                # print("??")
                continue
            if target.index not in graph.nodes():
                continue
            neighbor = list(graph[target.index].keys())
            # print('target', simplexs[cluster][0][nodes_to_index[cluster][0][target.index]].nodes)
            # print('neighbor', neighbor)
            # print('q', q)
            # print(n[2].index, n[2].train_node)
            # if len(list(graph.nodes())) == 522:
            #     print(target.index, neighbor)
            if len(neighbor) == 0:  # 可能出现无法继续消除的情况, 到极限就退出
                # print("??!")
                continue
            elif len(neighbor) > 1:
                if not reduce_edge:
                    print("Can't go forward.")
                    break
                # if (not flag_none[cluster]) and reduced_rank[cluster] < max_rank:
                #     reduced_rank[cluster] += 1
                #     graphs[cluster], flag_none[cluster], simplexs[cluster], maximal_cliques[cluster], nodes_to_index[cluster], reduced_edge_list, cnts[cluster] = Graph_reduce(graphs[cluster], 1, simplexs[cluster], nodes_to_index[cluster])
                #     # print('%d list_len: ' % reduced_rank[cluster], len(reduced_edge_list))
                #     for i, j in reduced_edge_list:
                #         n[i].edgenode -= 1
                #         n[i].vanished += 1
                #         n[j].edgenode -= 1
                #         # print('putting', n[i].index, n[i].edgenode, n[j].index, n[j].edgenode)
                #         q[cluster].put(copy.deepcopy(n[i]))
                #         q[cluster].put(copy.deepcopy(n[j]))
                #     # print('putting', n[target.index].index, n[target.index].edgenode)
                #     q[cluster].put(copy.deepcopy(n[target.index]))
                #     graphs[cluster] = graph
                #     continue

                if len(neighbor) == 2 and target.recast == 0:
                    n1 = min(neighbor[0], neighbor[1])
                    n2 = max(neighbor[0], neighbor[1])
                    # print(target.index, n1, n2)
                    # print("reducing", n1, list(graph.neighbors(n1)), simplexs[cluster][0][nodes_to_index[cluster][0][n1]].vanished, simplexs[cluster][0][nodes_to_index[cluster][0][n1]].num_maximum_face, [maximal_cliques[cluster][cqc].nodes for cqc in list(simplexs[cluster][0][nodes_to_index[cluster][0][n1]].maximum_face)])
                    # print("reducing", n2, list(graph.neighbors(n2)), simplexs[cluster][0][nodes_to_index[cluster][0][n2]].vanished, simplexs[cluster][0][nodes_to_index[cluster][0][n2]].num_maximum_face, [maximal_cliques[cluster][cqc].nodes for cqc in list(simplexs[cluster][0][nodes_to_index[cluster][0][n2]].maximum_face)])
                    idx_tar_1 = 0
                    idx_tar_ma = 0
                    neighbors = list(np.intersect1d(list(graph.neighbors(n2)), list(graph.neighbors(n1))))
                    if n1 not in graph.neighbors(n2) and len(neighbors) == 1 and (not n[target.index].train_node) and (
                            not (n[n1].train_node and n[n2].train_node)):
                        # print(target.index)
                        # qq = Queue()
                        graph.remove_node(target.index)
                        # 删除 simplexs 中(target.index, n1)对应的单纯形
                        nodes = [min(target.index, n1), max(target.index, n1)]
                        idx_tar_1 = nodes_to_index[cluster][1][tuple(nodes)]
                        idx_tar_ma = nodes_to_index[cluster]["max"][tuple(nodes)]
                        idx_1 = nodes_to_index[cluster][1][tuple(nodes)]
                        simplexs[cluster][1][idx_1].vanished = True
                        for other_node in simplexs[cluster][1][idx_1].low:
                            simplexs[cluster][0][other_node].high.discard(idx_1)
                        # 更新其所有的face对应的maximum face的数量
                        idx_max = nodes_to_index[cluster]["max"][tuple(nodes)]
                        maximal_cliques[cluster][idx_max].vanished = True
                        for pair in maximal_cliques[cluster][idx_max].face:
                            simplexs[cluster][pair[0]][pair[1]].num_maximum_face -= 1
                            # if pair[0] == 1 and simplexs[cluster][pair[0]][pair[1]].num_maximum_face == 1:
                            #     qq.put(simplexs[pair[0]][pair[1]].nodes[0])
                            simplexs[cluster][pair[0]][pair[1]].maximum_face.discard(idx_max)
                            # if len(simplexs[cluster][pair[0]][pair[1]].maximum_face) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #     print("14!!!")
                            # if pair[0] == 0:
                            #     if len([c for c in nx.find_cliques(graph) if pair[1] in c]) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #         print("14!!!")
                            # if pair[0] == 1:
                            #     if len([c for c in nx.find_cliques(graph) if (simplexs[pair[0]][pair[1]].nodes[0] in c and simplexs[pair[0]][pair[1]].nodes[1] in c)]) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #         print("14!!!")

                        # 删除 simplexs 中(target.index, n2)对应的单纯形
                        nodes = [min(target.index, n2), max(target.index, n2)]
                        idx_1 = nodes_to_index[cluster][1][tuple(nodes)]
                        simplexs[cluster][1][idx_1].vanished = True
                        for other_node in simplexs[cluster][1][idx_1].low:
                            simplexs[cluster][0][other_node].high.discard(idx_1)
                        # 更新其所有的face对应的maximum face的数量
                        idx_max = nodes_to_index[cluster]["max"][tuple(nodes)]
                        maximal_cliques[cluster][idx_max].vanished = True
                        for pair in maximal_cliques[cluster][idx_max].face:
                            simplexs[cluster][pair[0]][pair[1]].num_maximum_face -= 1
                            # if pair[0] == 1 and simplexs[cluster][pair[0]][pair[1]].num_maximum_face == 1:
                            #     q.put(pair[1])
                            simplexs[cluster][pair[0]][pair[1]].maximum_face.discard(idx_max)
                            # if len(simplexs[cluster][pair[0]][pair[1]].maximum_face) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #     print("15!!!")
                            # if pair[0] == 0:
                            #     if len([c for c in nx.find_cliques(graph) if pair[1] in c]) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #         print("15!!!")
                            # if pair[0] == 1:
                            #     if len([c for c in nx.find_cliques(graph) if (simplexs[cluster][pair[0]][pair[1]].nodes[0] in c and simplexs[cluster][pair[0]][pair[1]].nodes[1] in c)]) != simplexs[cluster][pair[0]][pair[1]].num_maximum_face:
                            #         print("15!!!")

                        graph.add_edge(n1, n2)
                        simplexs[cluster][1][idx_tar_1] = simplex(idx_tar_1, 1)
                        simplexs[cluster][0][n1].high.add(idx_tar_1)
                        simplexs[cluster][0][n2].high.add(idx_tar_1)
                        simplexs[cluster][1][idx_tar_1].low.add(n1)
                        simplexs[cluster][1][idx_tar_1].low.add(n2)
                        simplexs[cluster][1][idx_tar_1].nodes = [n1, n2]
                        nodes_to_index[cluster][1][(n1, n2)] = idx_tar_1

                        ll = [n1, n2]
                        numnode = 2
                        if numnode > 1:
                            key = (n1, n2)
                            maximal_cliques[cluster][idx_tar_ma] = simplex(idx_tar_ma, numnode - 1)
                            maximal_cliques[cluster][idx_tar_ma].nodes = [n1, n2]
                            nodes_to_index[cluster]["max"][key] = idx_tar_ma
                            for i in range(numnode):
                                maximal_cliques[cluster][idx_tar_ma].face.add((0, nodes_to_index[cluster][0][ll[i]]))
                                # print(ll[i])
                                # print(simplexs[0][ll[i]])
                                simplexs[cluster][0][ll[i]].maximum_face.add(idx_tar_ma)
                                simplexs[cluster][0][ll[i]].num_maximum_face += 1
                                # if len([c for c in nx.find_cliques(graph) if ll[i] in c]) != simplexs[cluster][0][ll[i]].num_maximum_face:
                                #     print("16!!!")
                        # print("check", n1, n2, nodes_to_index[cluster][1][(n1, n2)])

                        less_one = n1
                        if not n[target.index].train_node:
                            if n[less_one].train_node:
                                less_one = n2
                            elif (not n[less_one].train_node) and n[n2].vanished < n[n1].vanished:
                                less_one = n2
                        else:
                            if n[n1].train_node or ((not n[n2].train_node) and n[n2].vanished < n[n1].vanished):
                                less_one = n2
                        # print(n[less_one].train_node, n[target.index].train_node)
                        n[less_one].nodes = n[less_one].nodes + target.nodes + [target.index]
                        n[less_one].ed_van = n[less_one].ed_van + target.ed_van
                        # n[target.index].edgenode -= 1
                        # n[less_one].edgenode -= 1
                        n[less_one].train_node = (n[less_one].train_node or n[target.index].train_node)
                        n[less_one].remain = (n[less_one].remain or n[target.index].remain)
                        n[less_one].vanished += target.vanished
                        # print('putting', n[neighbor[0]].index, n[neighbor[0]].edgenode)
                        q[cluster].put(copy.deepcopy(n[less_one]))
                        n_vanished += 1
                        node_vanished[target.index] = 1
                        graphs[cluster] = graph
                        if n_vanished >= target_n:
                            res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph,
                                                                      node_to_cluster, back_edges, percentage,
                                                                      max_cluster_size)
                            pos_r = ratio_list.index(percentage)
                            if pos_r != len(ratio_list) - 1:
                                percentage = ratio_list[pos_r + 1]
                                target_n = len(data.x) * percentage
                            else:
                                return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges

                        # if 146118 in graphs[cluster].nodes():
                        #     print("shrink chain")
                        #     print("check_maximal", [c for c in nx.find_cliques(graphs[cluster]) if 146118 in c])
                        #     print("num_maximal", simplexs[cluster][0][146118].num_maximum_face)
                        # print("after", n1, list(graph.neighbors(n1)), simplexs[cluster][0][nodes_to_index[cluster][0][n1]].vanished, simplexs[cluster][0][nodes_to_index[cluster][0][n1]].num_maximum_face, [maximal_cliques[cluster][cqc].nodes for cqc in list(simplexs[cluster][0][nodes_to_index[cluster][0][n1]].maximum_face)])
                        # print("after", n2, list(graph.neighbors(n2)), simplexs[cluster][0][nodes_to_index[cluster][0][n2]].vanished, simplexs[cluster][0][nodes_to_index[cluster][0][n2]].num_maximum_face, [maximal_cliques[cluster][cqc].nodes for cqc in list(simplexs[cluster][0][nodes_to_index[cluster][0][n2]].maximum_face)])
                        continue
                    else:
                        n[target.index].recast += 1
                        q[cluster].put(copy.deepcopy(n[target.index]))
                        graphs[cluster] = graph
                        continue
                if edge_vanishing_cnt % 1000 == 0:
                    print('reduce edge: ', edge_vanishing_cnt)

                edge_vanishing_cnt += 1
                # print('Do edge vanishing:', edge_vanishing_cnt)
                q[cluster].put(copy.deepcopy(n[target.index]))
                n1 = 0
                n2 = 0
                prob = 0.2
                # print(target.index, target.edgenode,n[target.index].index, n[target.index].edgenode, list(graphs[cluster].neighbors(target.index)))
                edge_list = list(graphs[cluster].edges())
                target_edge = random.choice(edge_list)
                bridge_list = list(nx.bridges(graphs[cluster]))
                # print(edge_list, bridge_list)
                edge_left = list(set(edge_list) - set(bridge_list))
                # print(edge_left)
                if len(edge_left) == 0:
                    target_edge = random.choice(edge_list)
                else:
                    target_edge = random.choice(edge_left)
                target = n[target_edge[0]]
                node_2 = target_edge[1]
                n1 = min(target.index, node_2)
                n2 = max(target.index, node_2)
                back_edges.append((target.index, node_2))
                # num_triangles = len(simplexs[cluster][1][nodes_to_index[cluster][1][(n1, n2)]].high)
                # if num_triangles > 6:
                #     prob = 1
                # while random.random() > prob:
                #     target_edge = random.choice(list(graphs[cluster].edges()))
                #     target = n[target_edge[0]]
                #     node_2 = target_edge[1]
                #     n1 = min(target.index, node_2)
                #     n2 = max(target.index, node_2)
                #     num_triangles = len(simplexs[cluster][1][nodes_to_index[cluster][1][(n1, n2)]].high)
                #     if num_triangles > 6:
                #         prob = 1

                # node_2 = neighbor[-1]
                # graph.remove_edge(target.index, node_2)

                # if simplexs[cluster][1][nodes_to_index[cluster][1][(n1, n2)]].vanished:
                #     print('something wrong here')
                reduced_edge_list = []
                reduced_node_list = []
                simplex_list = []
                # print("reducing edge: ", n1, n2)
                # print("    node1", n1, neighbor, cnts[cluster], simplexs[cluster][0][n1].maximum_face, n[n1].edgenode,
                #       n[n1].edgenode, graph)
                # for key in simplexs[cluster][0][n1].maximum_face:
                #     print("        maximal_clique", maximal_cliques[cluster][key].nodes)
                # print("    node2", n2, neighbor, cnts[cluster], simplexs[cluster][0][n2].maximum_face, n[n2].edgenode,
                #       n[n2].edgenode, graph)
                # for key in simplexs[cluster][0][n2].maximum_face:
                #     print("        maximal_clique", maximal_cliques[cluster][key].nodes)

                if num_cluster > 1:
                    graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                        cluster], reduced_node_list, reduced_edge_list, back_edges, simplex_list = edge_collapse(graph,
                                                                                                                 [n1,
                                                                                                                  n2],
                                                                                                                 simplexs[
                                                                                                                     cluster],
                                                                                                                 nodes_to_index[
                                                                                                                     cluster],
                                                                                                                 maximal_cliques[
                                                                                                                     cluster],
                                                                                                                 cnts[
                                                                                                                     cluster],
                                                                                                                 [],
                                                                                                                 reduced_edge_list,
                                                                                                                 max_cluster_size,
                                                                                                                 recoil_percent,
                                                                                                                 back_edges,
                                                                                                                 simplex_list)
                    graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                        cluster], reduced_node_list, reduced_edge_list, back_edges = collapse_other_simplex(
                        simplex_list, graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster],
                        cnts[cluster], reduced_node_list, reduced_edge_list,
                        max_cluster_size, recoil_percent, back_edges)
                else:
                    graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                        cluster], reduced_node_list, reduced_edge_list, back_edges, simplex_list = edge_collapse(graph,
                                                                                                                 [n1,
                                                                                                                  n2],
                                                                                                                 simplexs[
                                                                                                                     cluster],
                                                                                                                 nodes_to_index[
                                                                                                                     cluster],
                                                                                                                 maximal_cliques[
                                                                                                                     cluster],
                                                                                                                 cnts[
                                                                                                                     cluster],
                                                                                                                 [],
                                                                                                                 reduced_edge_list,
                                                                                                                 max_cluster_size,
                                                                                                                 0,
                                                                                                                 back_edges,
                                                                                                                 simplex_list)
                    graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                        cluster], reduced_node_list, reduced_edge_list, back_edges = collapse_other_simplex(
                        simplex_list, graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster],
                        cnts[cluster], reduced_node_list, reduced_edge_list,
                        max_cluster_size, recoil_percent, back_edges)
                # graph, simplexs[cluster], nodes_to_index[cluster], reduced_edge_list = reduce(graph, , simplexs[cluster], nodes_to_index[cluster])

                for i, j in reduced_edge_list:
                    n[i].edgenode -= 1
                    n[i].vanished += 1
                    q[cluster].put(copy.deepcopy(n[i]))
                    if j != -1:
                        n[j].edgenode -= 1
                        q[cluster].put(copy.deepcopy(n[j]))

                for pair in reduced_node_list:
                    if pair[1] == -2:
                        n[pair[0]].recast = 0
                    else:
                        n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                        # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                        n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                        n[pair[1]].recast = 0
                        if n[pair[1]].label != n[pair[0]].label:
                            n[pair[1]].label = -1
                        # n[n[pair[0]].index].edgenode -= 1
                        # n[pair[1]].edgenode -= 1
                        n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                        n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                        n[pair[1]].vanished += n[pair[0]].vanished
                        q[cluster].put(copy.deepcopy(n[pair[1]]))
                        n_vanished += 1
                        node_vanished[n[pair[0]].index] = 1
                if n_vanished >= target_n:
                    res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph, node_to_cluster,
                                                              back_edges, percentage, max_cluster_size)
                    pos_r = ratio_list.index(percentage)
                    if pos_r != len(ratio_list) - 1:
                        pesrcentage = ratio_list[pos_r + 1]
                        target_n = len(data.x) * percentage
                    else:
                        return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges

                # if len(reduced_edge_list) > 1:
                #     print("Extra: %d edges." % len(reduced_edge_list))

                # n[target.index].edgenode -= 1
                # n[target.index].vanished += 1
                # n[node_2].edgenode -= 1
                n[target.index].ed_van.append(target.index)
                n[target.index].ed_van.append(node_2)
                # print('putting', n[target.index].index, n[target.index].edgenode, n[node_2].index, n[node_2].edgenode)
                # sub = nodes(n[node_2].index)
                # sub.edgenode
                # q[cluster].put(copy.deepcopy(n[node_2]))
                # q[cluster].put(copy.deepcopy(n[target.index]))
                # print('edges', node_2, target.index)
                # print('edgenode', n[target.index].edgenode)
                # print("    after node1", n1, neighbor, cnts[cluster], simplexs[cluster][0][n1].maximum_face,
                #       n[n1].edgenode, n[n1].edgenode, graph)
                # for key in simplexs[cluster][0][n1].maximum_face:
                #     print("        maximal_clique", maximal_cliques[cluster][key].nodes)
                # print("    after node2", n2, neighbor, cnts[cluster], simplexs[cluster][0][n2].maximum_face,
                #       n[n2].edgenode, n[n2].edgenode, graph)
                # for key in simplexs[cluster][0][n2].maximum_face:
                #     print("        maximal_clique", maximal_cliques[cluster][key].nodes)
                graphs[cluster] = graph
                continue
            # print(n[neighbor[0]].index, n[neighbor[0]].train_node)
            # print(n[2].index, n[2].train_node)
            # print(target.index, target.train_node)
            # sim = F.cosine_similarity(data.x[neighbor[0]], data.x[target.index], dim=0)
            # if n[neighbor[0]].train_node or n[target.index].train_node:
            #     # # print('two trian')
            #     # if sim < high_sim:
            #     continue
            #     n1 = min(target.index, neighbor[0])
            #     n2 = max(target.index, neighbor[0])
            #     n[target.index].edgenode -= 1
            #     n[neighbor[0]].edgenode -= 1
            #     # print('putting', n[neighbor[0]].index, n[neighbor[0]].edgenode)
            #     q[cluster].put(copy.deepcopy(n[neighbor[0]]))
            #     reduced_edge_list = []
            #     graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[cluster], reduced_edge_list = edge_collapse(graph, [n1, n2], simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[cluster], reduced_edge_list)
            #     # print('removing', target.index)
            #     for i, j in reduced_edge_list:
            #         n[i].edgenode -= 1
            #         n[i].vanished += 1
            #         n[j].edgenode -= 1
            #         # print('putting', n[i].index, n[i].edgenode, n[j].index, n[j].edgenode)
            #         q[cluster].put(copy.deepcopy(n[i]))
            #         q[cluster].put(copy.deepcopy(n[j]))
            #     if len(reduced_edge_list) > 1:
            #         print("Extra: %d edges." % len(reduced_edge_list))
            #     back_edges.append((neighbor[0], target.index))
            #     graphs[cluster] = graph
            # else:
            #     if sim < low_sim:
            #         n1 = min(target.index, neighbor[0])
            #         n2 = max(target.index, neighbor[0])
            #         n[target.index].edgenode -= 1
            #         n[neighbor[0]].edgenode -= 1
            #         # print('putting', n[neighbor[0]].index, n[neighbor[0]].edgenode)
            #         q[cluster].put(copy.deepcopy(n[neighbor[0]]))
            #         reduced_edge_list = []
            #         graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[cluster], reduced_edge_list = edge_collapse(graph, [n1, n2], simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[cluster], reduced_edge_list)
            #         # print('removing', target.index)
            #         for i, j in reduced_edge_list:
            #             n[i].edgenode -= 1
            #             n[i].vanished += 1
            #             n[j].edgenode -= 1
            #             # print('putting', n[i].index, n[i].edgenode, n[j].index, n[j].edgenode)
            #             q[cluster].put(copy.deepcopy(n[i]))
            #             q[cluster].put(copy.deepcopy(n[j]))
            #         if len(reduced_edge_list) > 1:
            #             print("Extra: %d edges." % len(reduced_edge_list))
            #         back_edges.append((neighbor[0], target.index))
            #         graphs[cluster] = graph
            #         continue
            #     else:
            q2 = Queue()
            simplex_list = []
            reduced_edge_list = []
            reduced_node_list = []
            # print(4, len(neighbor))

            print(target.index, neighbor, cnts[cluster], simplexs[cluster][0][target.index].maximum_face,
                  target.edgenode, n[target.index].edgenode, graph)

            # if 146118 in graphs[cluster].nodes():
            #     print("CHECK 3", simplexs[cluster].keys())
            #     if 0 in simplexs[cluster].keys():
            #         print("0 size", len(list(simplexs[cluster][0].keys())))
            #         if 146118 in simplexs[cluster][0].keys():
            #             print("maximum_face", list(simplexs[cluster][0][146118].maximum_face))
            if num_cluster > 1:
                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = node_collapse(graph,
                                                                                                                 target.index,
                                                                                                                 simplexs[
                                                                                                                     cluster],
                                                                                                                 nodes_to_index[
                                                                                                                     cluster],
                                                                                                                 maximal_cliques[
                                                                                                                     cluster],
                                                                                                                 cnts[
                                                                                                                     cluster],
                                                                                                                 [],
                                                                                                                 [], q2,
                                                                                                                 max_cluster_size,
                                                                                                                 recoil_percent,
                                                                                                                 back_edges,
                                                                                                                 simplex_list)

                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, back_edges, simplex_list = collapse_other_edge(q2,
                                                                                                                   graph,
                                                                                                                   simplexs[
                                                                                                                       cluster],
                                                                                                                   nodes_to_index[
                                                                                                                       cluster],
                                                                                                                   maximal_cliques[
                                                                                                                       cluster],
                                                                                                                   cnts[
                                                                                                                       cluster],
                                                                                                                   reduced_node_list,
                                                                                                                   reduced_edge_list,
                                                                                                                   max_cluster_size,
                                                                                                                   recoil_percent,
                                                                                                                   back_edges,
                                                                                                                   simplex_list)
                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, back_edges = collapse_other_simplex(
                    simplex_list, graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster],
                    cnts[cluster], reduced_node_list, reduced_edge_list,
                    max_cluster_size, recoil_percent, back_edges)
            else:

                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, q2, back_edges, simplex_list = node_collapse(graph,
                                                                                                                 target.index,
                                                                                                                 simplexs[
                                                                                                                     cluster],
                                                                                                                 nodes_to_index[
                                                                                                                     cluster],
                                                                                                                 maximal_cliques[
                                                                                                                     cluster],
                                                                                                                 cnts[
                                                                                                                     cluster],
                                                                                                                 [],
                                                                                                                 [], q2,
                                                                                                                 max_cluster_size,
                                                                                                                 0,
                                                                                                                 back_edges,
                                                                                                                 simplex_list)

                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, back_edges, simplex_list = collapse_other_edge(q2,
                                                                                                                   graph,
                                                                                                                   simplexs[
                                                                                                                       cluster],
                                                                                                                   nodes_to_index[
                                                                                                                       cluster],
                                                                                                                   maximal_cliques[
                                                                                                                       cluster],
                                                                                                                   cnts[
                                                                                                                       cluster],
                                                                                                                   reduced_node_list,
                                                                                                                   reduced_edge_list,
                                                                                                                   max_cluster_size,
                                                                                                                   0,
                                                                                                                   back_edges,
                                                                                                                   simplex_list)
                graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster], cnts[
                    cluster], reduced_node_list, reduced_edge_list, back_edges = collapse_other_simplex(
                    simplex_list, graph, simplexs[cluster], nodes_to_index[cluster], maximal_cliques[cluster],
                    cnts[cluster], reduced_node_list, reduced_edge_list,
                    max_cluster_size, 0, back_edges)
            for i, j in reduced_edge_list:
                n[i].edgenode -= 1
                n[i].vanished += 1
                q[cluster].put(copy.deepcopy(n[i]))
                if j != -1:
                    n[j].edgenode -= 1
                    q[cluster].put(copy.deepcopy(n[j]))

            for pair in reduced_node_list:
                if pair[1] == -2:
                    n[pair[0]].recast = 0
                else:
                    n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                    # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                    n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                    n[pair[1]].recast = 0
                    # n[n[pair[0]].index].edgenode -= 1
                    # n[pair[1]].edgenode -= 1
                    if n[pair[1]].label != n[pair[0]].label:
                        n[pair[1]].label = -1
                    n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                    n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                    n[pair[1]].vanished += n[pair[0]].vanished
                    q[cluster].put(copy.deepcopy(n[pair[1]]))
                    n_vanished += 1
                    node_vanished[n[pair[0]].index] = 1
            graphs[cluster] = graph
            if n_vanished >= target_n:
                res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph, node_to_cluster,
                                                          back_edges, percentage, max_cluster_size)
                pos_r = ratio_list.index(percentage)
                if pos_r != len(ratio_list) - 1:
                    percentage = ratio_list[pos_r + 1]
                    target_n = len(data.x) * percentage
                else:
                    return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges

    end = time.time_ns()
    # global maximun_memory
    pid = os.getpid()

    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory = memory_info.rss
    if memory - current_memory > maximun_memory:
        maximun_memory = memory - current_memory
        # print('3 Memory cost:', memory-current_memory)

    # 创建一个记录内存使用的列表

    f = open('time_record_1000', 'a')
    f.write('Cora, %ld\n' % ((end - start) // 1000000))
    f.close()

    # res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, old_graph, node_to_cluster, back_edges, percentage, max_cluster_size)
    # print(old_to_new)
    return data, {}, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


from scipy.sparse import find
# from utils import load_reddit_data, sgc_precompute, set_seed, loadRedditFromNPZ

recoil_percent = 0.5
ratio_list = [0.5, 0.7, 0.8, 0.9]
percentage = ratio_list[0]
dataname = 'Cora'
if dataname == 'dblp':
    dataset = CitationFull(root='/home/ycmeng/ep1/dataset', name=dataname)
elif dataname == 'Physics':
    dataset = Coauthor(root='/home/ycmeng/ep1/dataset', name=dataname)
elif dataname == 'ogbn-products':
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/mnt/ssd2/products/raw')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator('ogbn-products')
elif dataname == 'OGBN-arxiv':
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator('ogbn-arxiv')
elif dataname == 'reddit':
    dataset = Reddit(root='/home/ycmeng/ep1/dataset')
elif dataname == 'reddit2':
    # dataset = Reddit(root='/home/ycmeng/ep1/dataset')
    data, _ = np.load('./Reduced_Node_Data/reddit2.npy', allow_pickle=True)

    # data = dataset[0]
    # adj, train_adj, features, labels, train_index, val_index, test_index = load_reddit_data(data_path="/home/ycmeng/ep1/dataset/reddit2/raw/", normalization="AugNormAdj")
    # num_node = adj.shape[0]
    # l1 = []
    # l2 = []
    # edges = {}
    # cnt = 0
    # # 将图之间的边加上
    # print(type(adj))
    # row, col = adj.nonzero()
    # row = list(row)
    # col = list(col)
    # for i in range(len(row)):
    #     node1 = row[i]
    #     node2 = col[i]
    #     if (min(node1, node2), max(node1, node2)) not in edges.keys():
    #         l1.append(node1)
    #         l2.append(node2)
    #         l1.append(node2)
    #         l2.append(node1)
    #         edges[(min(node1, node2), max(node1, node2))] = 1
    #         cnt+=1

    # # for node1 in range(num_node):
    # #     row = (adj.getrow(node1)).toarray()
    # #     if node1 % 1000 == 0:
    # #         print(node1)
    # #     for node2 in range(num_node):
    # #         # print(adj)
    # #         if row[0][node2] == 1:
    # #             l1.append(node1)
    # #             l2.append(node2)
    # #             l1.append(node2)
    # #             l2.append(node1)
    # #             edges[(min(node1, node2), max(node1, node2))] = 1
    # #             cnt+=1
    # data.edge_index = torch.cat((torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0)), 0)
    # data.train_mask = index_to_mask(torch.tensor(train_index), size=num_node )
    # data.val_mask = index_to_mask(torch.tensor(val_index), size=num_node )
    # data.test_mask = index_to_mask(torch.tensor(test_index), size=num_node )
    # data.x = torch.tensor(features)
    # data.y = labels
    # print(data)
    # m = {}
    # np.save('./Reduced_Node_Data/reddit2.npy', (data, m))
    # print("saved")
    # print(adj.shape, features.shape, y_train.shape, y_val.shape, y_test.shape, train_index.shape, val_index.shape, test_index.shape)
elif dataname == 'pubmed':
    dataset = Planetoid(root='/home/ycmeng/ep1/dataset', name=dataname)
else:
    dataset = Planetoid(root='/home/ycmeng/ep1/', name=dataname)

# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = './')

high_sim = 2
low_sim = 0.0
print(time.time())

device = torch.device('cpu')
if dataname != "reddit2":
    data = dataset[0].to(device)
    print(dataset)
print(data)
target_n = len(data.x) * percentage
if dataname == 'OGBN-arxiv' or dataname == 'ogbn-products':
    split_idx = dataset.get_idx_split()
    data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
    data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
    data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
    data.y = torch.squeeze(data.y)
if dataname == 'dblp' or dataname == 'Physics':
    indices = []
    num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    test_index = torch.cat([i[50:] for i in indices], dim=0)
    print(data.num_nodes)
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
print(torch.unique(data.y, return_counts=True))
print(torch.unique(data.y[data.train_mask], return_counts=True))
print(torch.sum(torch.unique(data.y[data.train_mask], return_counts=True)[1]))

keep_ratio = 1.0
indices = []
num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
index_train = (data.train_mask == 1).nonzero().view(-1)
for i in range(num_classes):
    index = (data.y == i).nonzero().view(-1)
    tensor_isin = torch.isin(index, index_train)
    index = index[tensor_isin]
    index = index[torch.randperm(index.size(0))]
    indices.append(index)

keep_index = torch.cat([i[:int(i.size()[0] * keep_ratio)] for i in indices], dim=0)
keep_mask = index_to_mask(keep_index, size=data.num_nodes)
print(torch.unique(data.y[keep_mask], return_counts=True))
# data.x = torch.zeros([6, 1433])
# data.edge_index = torch.IntTensor([[1, 2], [1, 3], [1, 4], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [1, 0], [2, 0], [3, 0], [5, 0]]).t()

# data.x = torch.zeros([6, 1433])
# data.edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [0, 1]]).t()

# data.x = torch.zeros([9, 1433])
# llll = []
# llll.append([1, 2])
# llll.append([1, 3])
# llll.append([1, 4])
# llll.append([2, 3])
# llll.append([2, 4])
# llll.append([3, 4])
# llll.append([3, 5])
# llll.append([4, 5])
# llll.append([4, 7])
# llll.append([5, 6])
# llll.append([6, 7])
# llll.append([7, 8])
# llll.append([8, 0])
# data.edge_index = torch.IntTensor(llll).t()

# data.x = torch.zeros([100, 1433])
# llll = []
# for i in range(100):
#     for j in range(i+1, 100):
#         llll.append([i, j])
# data.edge_index = torch.IntTensor(llll).t()

# print(data)
# print(split_idx)
# train_mask = torch.zeros(169343)
# for i in split_idx['train']:
#     train_mask[int(i)] = 1
# val_mask = torch.zeros(169343)
# for i in split_idx['valid']:
#     val_mask[int(i)] = 1
# test_mask = torch.zeros(169343)
# for i in split_idx['test']:
#     test_mask[int(i)] = 1
# data.train_mask = train_mask
# data.val_mask = val_mask
# data.test_mask = test_mask
print(data)
# data.x = torch.FloatTensor(data.x)
# data = data.to(device)


import psutil
import os
import time

# 获取当前进程ID
pid = os.getpid()

# 创建一个记录内存使用的列表
memory_usage = []
back_edges = []
process = psutil.Process(pid)
memory_info = process.memory_info()
current_memory = memory_info.rss
# print(current_memory)
target_n_list = [int(num * len(data.x)) for num in ratio_list]
for max_cluster_size in [1000]:
    n = []
    graph = nx.Graph()
    for i in range(len(data.x)):
        graph.add_node(i)
        n.append(nodes(i))
        if keep_mask[i] == 1:
            n[i].train_node = True
            n[i].remain = True
        # if data.train_mask[i] == 1:
        #     n[i].label = int(data.y[i])
            # n[i].remain = True
        # if data.val_mask[i] == 1:
        # if random.random() > 0.5:
        # n[i].train_node = True
        # n[i].remain = True
        #     print('train_node', i)
    # n[2].train_node = False
    # n[3].train_node = False
    # n[0].train_node = False
    # n[5].train_node = False
    # print(graph.nodes())
    for i in range(len(data.edge_index[0])):
        n1n1 = int(data.edge_index[0, i])
        n2n2 = int(data.edge_index[1, i])
        if n1n1 != n2n2:
            graph.add_edge(n1n1, n2n2)
        # if int(data.edge_index[0, i]) == 0 or int(data.edge_index[1, i]) == 0:
        #     print("???")
        # n[int(data.edge_index[0, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
        # n[int(data.edge_index[1, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
    # print(graph.nodes())
    # print(graph.edges())
    cnt_cluster_node = 0
    num_nodes = data.x.size()[0]
    n_vanished = 0

    print("time:", time.time_ns())
    com_nodes = []
    components = nx.connected_components(graph)
    print(components)
    for component in components:
        com = list(component)
        # print("components", len(com), com)
        if len(com) >= 10:
            com_nodes = com_nodes + com
        else:
            add_flag = 0
            for node in com:
                if n[node].remain:
                    add_flag = 1
                    break
            if add_flag == 1:
                com_nodes = com_nodes + com
            else:
                for node in com:
                    n[node].vanished = True
                n_vanished += len(com)
    graph = nx.Graph(graph.subgraph(com_nodes))
    old_graph = nx.Graph(graph)
    # np.save('./origin_graph/ogbn-products.npy', (graph, n))
    # graph, n = np.load('./origin_graph/ogbn-products.npy', allow_pickle=True)
    print(graph)
    # graph = nx.Graph(graph)
    # n_vanished = 0
    cnt_cluster_node = 0
    num_nodes = data.x.size()[0]
    graph_save = copy.deepcopy(graph)
    node_to_cluster = list(0 for i in range(0, num_nodes))
    graphs = []
    cluster_node = []
    num_cluster = (len(graph.nodes()) - 1) // max_cluster_size + 1
    print(num_cluster)
    start = time.time_ns()
    # for i in range(1, num_cluster):
    #     # print("分图", i)
    #     cnt_cluster_node = 0
    #     current_nodes = []
    #     # print(max_cluster_size, cnt_cluster_node)
    #     while max_cluster_size > cnt_cluster_node:
    #         q1 = Queue()
    #         for node in graph.nodes():
    #             if not node_to_cluster[node]:
    #                 q1.put(node)
    #                 break
    #         while not q1.empty():
    #             node = q1.get()
    #             if node_to_cluster[node] != 0:
    #                 continue
    #             current_nodes.append(node)
    #             node_to_cluster[node] = i
    #             # print(current_nodes)
    #             cnt_cluster_node += 1
    #             neighbor = list(graph[node].keys())
    #             graph.remove_node(node)
    #             if max_cluster_size <= cnt_cluster_node:
    #                 break
    #             for node2 in neighbor:
    #                 if not node_to_cluster[node2]:
    #                     q1.put(node2)
    #         while not q1.empty():
    #             node = q1.get()
    #             if node_to_cluster[node] != 0:
    #                 continue
    #             if len(list(graph[node].keys())) == 0:
    #                 current_nodes.append(node)
    #                 node_to_cluster[node] = i
    #                 # print(current_nodes)
    #                 cnt_cluster_node += 1
    #                 graph.remove_node(node)
    #     # print(current_nodes)
    #     graphs.append(nx.Graph(graph_save.subgraph(current_nodes)))
    #     # print(graphs[i-1])
    #     cluster_node.append(current_nodes)
    # current_nodes = []
    # for node in graph.nodes():
    #     if not node_to_cluster[node]:
    #         current_nodes.append(node)
    # graphs.append(nx.Graph(graph_save.subgraph(current_nodes)))
    # cluster_node.append(current_nodes)
    # num_nodes = data.x.size()[0]
    # new_node_to_cluster = list(0 for i in range(0, num_nodes))
    # new_graphs = []
    # cnt_cluster_node = 0
    current_nodes = []
    i = 1
    # print(max_cluster_size, cnt_cluster_node)

    # 重新对得到的图进行子图切分
    q1 = Queue()
    for node in graph.nodes():
        # print(node, not new_node_to_cluster[node], len(list(graph[node].keys())))
        if (not node_to_cluster[node]) and len(list(graph[node].keys())) != 0:
            q1.put(copy.deepcopy(node))
            while not q1.empty():
                node = q1.get()
                # print('here is the node', node)
                if node_to_cluster[node] != 0:
                    continue
                    # print('here is the node', node)
                current_nodes.append(node)
                node_to_cluster[node] = i
                # print('update', node)
                for nei in n[node].nodes:
                    # print('update', nei)
                    node_to_cluster[nei] = i
                # print(current_nodes)
                cnt_cluster_node += 1
                if max_cluster_size <= cnt_cluster_node:
                    # while not q1.empty():
                    #     node = q1.get()
                    #     if node_to_cluster[node] != 0:
                    #         continue
                    #     if len(list(graph[node].keys())) == 0:
                    #         current_nodes.append(node)
                    #         node_to_cluster[node] = i
                    #         # print(current_nodes)
                    #         cnt_cluster_node += 1
                    #         graph.remove_node(node)
                    # print(i)
                    q1 = Queue()
                    i += 1
                    graphs.append(nx.Graph(graph.subgraph(current_nodes)))
                    cluster_node.append(current_nodes)
                    current_nodes = []
                    cnt_cluster_node = 0
                    break
                neighbor = list(graph[node].keys())
                for node2 in neighbor:
                    if not node_to_cluster[node2]:
                        q1.put(copy.deepcopy(node2))
    # print(current_nodes)
    # print(new_node_to_cluster)
    for node in graph.nodes():
        if not node_to_cluster[node]:
            current_nodes.append(node)
            node_to_cluster[node] = i
            # print('node', node)
            # print(n[node].nodes)
            for nei in n[node].nodes:
                node_to_cluster[nei] = i
            # print(current_nodes)
            cnt_cluster_node += 1
            if max_cluster_size <= cnt_cluster_node:
                graphs.append(nx.Graph(graph.subgraph(current_nodes)))
                cluster_node.append(current_nodes)
                i += 1
                cnt_cluster_node = 0
                current_nodes = []
    if len(current_nodes) != 0:
        graphs.append(nx.Graph(graph.subgraph(current_nodes)))
        cluster_node.append(current_nodes)

    num_cluster = len(graphs)
    # for i in range(len(graphs)):
    #     print(graphs[i])
    graph = graph_save
    flag_none = list(0 for i in range(0, num_cluster))
    # 将节点映射到单纯性
    nodes_to_index = list({} for i in range(0, num_cluster))
    reduced_rank = list(1 for i in range(0, num_cluster))
    simplexs = list({} for i in range(0, num_cluster))
    maximal_cliques = list({} for i in range(0, num_cluster))
    cnts = list(0 for i in range(0, num_cluster))
    edge_vanishing_cnt = 0
    cluster_n_nanished = list(0 for i in range(0, num_cluster))
    q = list(PriorityQueue() for i in range(0, num_cluster))
    finished = list(0 for i in range(0, num_cluster))
    node_vanished = {}  # 已经被删除的点
    if num_cluster > 1:
        for i in range(num_cluster):
            # print("压图", i)
            # print(nx.is_frozen(graphs[i]))
            # print(i)
            reduced_node_list = []
            reduced_edge_list = []
            graphs[i], flag_none[i], simplexs[i], maximal_cliques[i], nodes_to_index[
                i], reduced_node_list, reduced_edge_list, cnts[i], back_edges = Graph_reduce(graphs[i], 6, simplexs[i],
                                                                                             nodes_to_index[i],
                                                                                             max_cluster_size,
                                                                                             recoil_percent, back_edges)

            for pair in reduced_node_list:
                if pair[1] == -2:
                    n[pair[0]].recast = 0
                else:
                    n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                    # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                    n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                    n[pair[1]].recast = 0
                    # n[n[pair[0]].index].edgenode -= 1
                    # n[pair[1]].edgenode -= 1
                    n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                    if n[pair[1]].label != n[pair[0]].label:
                        n[pair[1]].label = -1
                    n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                    n[pair[1]].vanished += n[pair[0]].vanished
                    n_vanished += 1
                    node_vanished[n[pair[0]].index] = 1

            if len(list(graphs[i].nodes())) <= max_cluster_size * recoil_percent and len(graphs) > 1:
                finished[i] = 1
                simplexs[i] = {}
                maximal_cliques[i] = {}
            if n_vanished >= target_n:
                res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, graph, node_to_cluster, [],
                                                          percentage, max_cluster_size)
                pos_r = ratio_list.index(percentage)
                if pos_r != len(ratio_list) - 1:
                    percentage = ratio_list[pos_r + 1]
                    target_n = len(data.x) * percentage
                else:
                    break

    else:
        for i in range(num_cluster):
            # print("压图", i)
            reduced_node_list = []
            reduced_edge_list = []
            graphs[i], flag_none[i], simplexs[i], maximal_cliques[i], nodes_to_index[
                i], reduced_node_list, reduced_edge_list, cnts[i], back_edges = Graph_reduce(graphs[i], 6, simplexs[i],
                                                                                             nodes_to_index[i],
                                                                                             max_cluster_size, 0,
                                                                                             back_edges)

            for pair in reduced_node_list:
                if pair[1] == -2:
                    n[pair[0]].recast = 0
                else:
                    n[pair[1]].nodes = n[pair[1]].nodes + n[pair[0]].nodes + [n[pair[0]].index]
                    # print(n[pair[0]].index, "to", n[pair[1]].index, " ", n[pair[0]].nodes, n[pair[1]].nodes)
                    n[pair[1]].ed_van = n[pair[1]].ed_van + n[pair[0]].ed_van
                    n[pair[1]].recast = 0
                    if n[pair[1]].label != n[pair[0]].label:
                        n[pair[1]].label = -1
                    # n[n[pair[0]].index].edgenode -= 1
                    # n[pair[1]].edgenode -= 1
                    n[pair[1]].train_node = (n[pair[1]].train_node or n[n[pair[0]].index].train_node)
                    n[pair[1]].remain = (n[pair[1]].remain or n[n[pair[0]].index].remain)
                    n[pair[1]].vanished += n[pair[0]].vanished
                    n_vanished += 1
                    node_vanished[n[pair[0]].index] = 1
            if n_vanished >= target_n:
                res, old_to_new, new_to_old = write_graph(data, num_cluster, graphs, n, graph, node_to_cluster, graph,
                                                          percentage, max_cluster_size)
                pos_r = ratio_list.index(percentage)
                if pos_r != len(ratio_list) - 1:
                    percentage = ratio_list[pos_r + 1]
                    target_n = len(data.x) * percentage

    # for i in range(num_cluster):
    #     print(simplexs[i].keys())
    #     for key in simplexs[i][1].keys():
    #         if simplexs[i][1][key].num_maximum_face == 1:
    #             print("strange")

    for i in range(num_cluster):
        for node in graphs[i].nodes():
            n[node].edgenode = 0
        for edge in graphs[i].edges():
            n[edge[0]].edgenode += 1
            n[edge[1]].edgenode += 1
    q = list(PriorityQueue() for i in range(0, num_cluster))

    for i in range(num_cluster):
        for node in graphs[i].nodes():
            if n[node].edgenode != 0:
                q[i].put(copy.deepcopy(n[node]))

    # for i in range(num_cluster):
    # print(nx.is_frozen(graphs[i]))
    node_vanished = {}  # 已经被删除的点
    # q = PriorityQueue()
    # for i in n:
    #     q.put(i)
    # flag_stop = 0
    # for i in range(num_cluster):
    #     for key in maximal_cliques[i]:
    #         if maximal_cliques[i][key].vanished == False and not check_maximal(graphs[i], maximal_cliques[i][key].nodes):
    #             print("nodes", maximal_cliques[i][key].nodes)
    #             print("????")
    #             flag_stop = 1
    #     cliques = find_max_cliques_nx(graphs[i], 100000)
    #     for clique in cliques:
    #         if tuple(clique) not in nodes_to_index[i]["max"]:
    #             print("lost_clique", clique)
    #             print("????")
    #             flag_stop = 1
    # color = list(0 for i in range(0, 2708))
    # now_color = 1
    # for i in range(2708):
    #     if color[i] != 0:
    #         continue
    #     q1 = Queue()
    #     q1.put(i)
    #     while not q1.empty():
    #         node = q1.get()
    #         color[node] = now_color
    #         neighbor = list(graphs[0][node].keys())
    #         for node2 in neighbor:
    #             if color[node2] == 0:
    #                 q1.put(node2)
    #     now_color += 1
    # print(now_color)
    end = time.time_ns()
    f = open('time_record_1000_record', 'a')
    f.write('Cora, %ld\n' % ((end - start) // 1000000))
    f.close()
    # for v in np.arange(0.00, 1, 0.05):
    print(type(data))
    for v in ratio_list:
        # for v in [0.9]:

        try:
            file = open('Reduce.txt', mode='a', encoding='utf-8')
        except:
            print(f'open analyse file error!')
        file.write('vanished: %lf' % v)
        file.write('\n')
        file.close()
        data_mol, m, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster, maximal_cliques, cnts, old_graph, back_edges = Modify(
            data, v, device, 6, True, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished,
            node_vanished, q, reduced_rank, num_cluster, finished, node_to_cluster, max_cluster_size, high_sim, low_sim,
            old_graph, maximal_cliques, cnts, recoil_percent, ratio_list, back_edges)
        data_mol = data_mol.to(device)
        # train_index = torch.zeros(data_mol.new_x.shape[0])

        # for i in range(int(data_mol.train_index.shape[0])):
        #     if m[i] = 1

        # np.save('./Reduced_Node_Data/%s_%.2f_split%d_check.npy' % (dataname, v, max_cluster_size), (data_mol.cpu(), m))
        # n_cnt = {}
        # for i in range(data_mol.new_edge.size()[1]):
        #     n_cnt[int(data_mol.new_edge[0][i])] = 1
        #     n_cnt[int(data_mol.new_edge[1][i])] = 1
        # print('len', len(n_cnt.keys()))
        # print(cnts)
        # print('edge_vanishing_cnt', edge_vanishing_cnt)
        # for i in simplexs[0].keys():
        #     print(i, len(simplexs[0][i].keys()))
        #     for idx in simplexs[0][i].keys():
        #         print(simplexs[0][i][idx].nodes, simplexs[0][i][idx].vanished)
        # print(data_mol)
        # print(data_mol.new_edge)
