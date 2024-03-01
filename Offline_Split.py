from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
import logging
import networkx as nx

import psutil
import os
import time
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
import random
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
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
warnings.filterwarnings('ignore')
# global maximun_memory
maximun_memory = 0
class Evaluator:
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
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
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

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
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
    

class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
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
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)


        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygNodePropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
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
            train_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)

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
        url =  self.meta_info['url']
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
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

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
            data = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip', header = None).values

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
    def __init__(self, index):            # 初始化 包含左端点所在的x, y坐标以及长度l
        self.index = index
        self.vanished = 0               # 这个点上重叠的点数
        self.edgenode = 0               # 这个点上的边数
        self.nodes = []                 # 这个点上重叠的点的列表
        self.ed_van = []                # 这个点上由于消去边获得的特征
        self.train_node = False         # 是否为训练节点

    def __lt__(self, other):                # 为了堆优化重载运算符"<"
        if self.edgenode < other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.vanished < other.vanished:
                return True
            return False

    def __gt__(self, other):                # 为了堆优化重载运算符">"
        if self.edgenode > other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.vanished > other.vanished:
                return True
            return False

    def __eq__(self, other):                # 为了堆优化重载运算符"="
        if self.edgenode == other.edgenode and self.vanished == other.vanished:
            return True
        else:
            return False
    

        
class simplex:
    def __init__(self, index, rank):            # 初始化 包含左端点所在的x, y坐标以及长度l
        self.index = index
        self.rank = rank
        self.nodes = []
        self.high = []
        self.low = []
        self.vanished = False


def Re_collapse(graph:nx.Graph, rank, simplexs, nodes_to_index, reduced_edge_list):
    cnt = 0
    for i in range(rank, 1, -1):
        # print('Reducing:', i)
        if i not in simplexs.keys():
            continue
        for idx, simp in simplexs[i].items():
            if simplexs[i][idx].vanished == True:
                continue
            if len(simplexs[i][idx].high) != 0:
                continue
            for coface in simp.low:
                if simplexs[i-1][coface].vanished == True:
                    continue
                if len(simplexs[i-1][coface].high) == 1:
                    cnt += 1
                    # 清除所有 i-1 维单纯形对该 i 维单纯形的记录 （即删除 i 维单纯形）
                    for other_coface in simp.low:
                        pos = simplexs[i-1][other_coface].high.index(idx)
                        simplexs[i-1][other_coface].high = simplexs[i-1][other_coface].high[:pos] + simplexs[i-1][other_coface].high[pos+1:]
                    # 清除所有 i-2 维单纯形对该 i-1 维单纯形的记录 （即删除 i-1 维单纯形）
                    for co_coface in simplexs[i-1][coface].low:
                        pos = simplexs[i-2][co_coface].high.index(coface)
                        simplexs[i-2][co_coface].high = simplexs[i-2][co_coface].high[:pos] + simplexs[i-2][co_coface].high[pos+1:]
                    # 如果 i-1 维单纯形是个边，那么同时删除它在 graph 上的记录
                    if i == 2:
                        graph.remove_edge(simplexs[i-1][coface].nodes[0], simplexs[i-1][coface].nodes[1])
                        reduced_edge_list.append((simplexs[i-1][coface].nodes[0], simplexs[i-1][coface].nodes[1]))
                    simplexs[i-1][coface].vanished = True
                    simplexs[i][idx].vanished = True
                    break
    # 考虑需要返回哪些数据用于后续。
    print('Need Collapse', cnt)
    return graph, simplexs, nodes_to_index, reduced_edge_list

def offline_build(graph, max_rank, simplexs, nodes_to_index, rank, nodes, coneighbor, new_node, num_facit, cnts, reduced_edge_list, less_cnts):
    global maximun_memory
    pid = os.getpid()

    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory = memory_info.rss
    if memory-current_memory > maximun_memory:
        maximun_memory = memory-current_memory
        print('Memory cost:', memory-current_memory)
    # 当前包含点
    nodes_new = nodes + [new_node]
    # print(nodes_new)
    # 当前的共同邻居
    coneighbor_new = list(np.intersect1d(coneighbor, list(graph[new_node].keys())))
    coneighbor_new.sort()
    # 确认其高维单纯形数量
    higher = len(coneighbor_new)
    # 用于帮助记录其高维单纯形的编号
    high = []
    # 需要考虑高维单纯性
    if rank < max_rank:
        # 对于所有的高位单纯形
        for node in coneighbor_new:
            # 这个高维单纯形已经被考虑过了
            if node < new_node:
                prob = nodes_new + [node]
                prob.sort()
                prob = tuple(prob)
                # 如果单纯形从未创建或已删除，更新高维单纯形数量
                if nodes_to_index[rank+1][prob] == -1 or simplexs[rank+1][nodes_to_index[rank+1][prob]].vanished:
                    higher -= 1
                # 如果该单纯形仍存在，则观察是否可以vanish
                else:
                    high_idx = nodes_to_index[rank+1][prob]
                    if rank >= 1 and higher == 1 and len(simplexs[rank+1][high_idx].high) == 0:
                        simplexs, graph, reduced_edge_list = clear_simplex(simplexs, simplexs[rank+1][high_idx], high_idx, rank+1, graph, reduced_edge_list)
                        nodes_to_index[rank][tuple(nodes_new)] = -1
                        if rank == 1:
                            graph.remove_edge(nodes_new[0], nodes_new[1])
                            reduced_edge_list.append((nodes_new[0], nodes_new[1]))
                        less_cnts[rank] += 1
                        return graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, False, True
                    high.append(node)
                        # simplexs[rank+1][high_idx].vanished = True
                        # for idx in simplexs[rank+1][high_idx].low:
                        #     pos = simplexs[rank][idx].high.index(high_idx)
                        #     simplexs[rank][idx].high = simplexs[rank][idx].high[:pos] + simplexs[rank][idx].high[pos+1:]
            else:
                # 未考虑的高维单纯形进行构建
                graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, vanish, high_vanish = offline_build(graph, max_rank, simplexs, nodes_to_index, rank + 1, nodes_new, coneighbor_new, node, higher, cnts, reduced_edge_list, less_cnts)
                # 如果当前单纯形需要继续 vanish , 直接返回
                if vanish:
                    nodes_to_index[rank][tuple(nodes_new)] = -1
                    if rank == 1:
                        graph.remove_edge(nodes_new[0], nodes_new[1])
                        reduced_edge_list.append((nodes_new[0], nodes_new[1]))
                    less_cnts[rank] += 1
                    return graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, False, True
                # 如果当前对应的高维单纯形被 vanish , 更新其对应的高维单纯形数量
                if high_vanish:
                    higher -= 1
                # 如果该高维单纯形没有被 vanish , 更新其所需的最后一个点
                else:
                    high.append(node)
    # 重新考虑这个单纯形是否可以进行进一步的 collapse
    # 当前单纯形可以被作为 maximum 单纯形进行 collapse
    if rank >= 2 and higher == 0 and num_facit == 1:
        nodes_to_index[rank][tuple(nodes_new)] = -1
        less_cnts[rank] += 1
        return graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, True, False
    # 该单纯形可能被作为 facet 参与 collapse
    elif rank >= 1 and higher == 1 and rank != max_rank:
        prob = nodes_new + [high[0]]
        prob.sort()
        prob = tuple(prob)
        high_idx = nodes_to_index[rank+1][prob]
        if len(simplexs[rank+1][high_idx].high) == 0:
            simplexs, graph, reduced_edge_list = clear_simplex(simplexs, simplexs[rank+1][high_idx], high_idx, rank+1, graph, reduced_edge_list)
            nodes_to_index[rank][tuple(nodes_new)] = -1
            if rank == 1:
                graph.remove_edge(nodes_new[0], nodes_new[1])
                reduced_edge_list.append((nodes_new[0], nodes_new[1]))
            less_cnts[rank] += 1
            return graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, False, True
    # 不需要继续构建更高维单纯性
    # 在返回前构造当前单纯形
    simplexs[rank][cnts[rank]] = simplex(cnts[rank], rank)
    simplexs[rank][cnts[rank]].nodes = nodes_new
    if rank+1 in nodes_to_index.keys():
        for node in coneighbor_new:
            prob = nodes_new + [node]
            prob.sort()
            prob = tuple(prob)
            high_idx = nodes_to_index[rank+1][prob]
            if high_idx == -1:
                continue
            simplexs[rank][cnts[rank]].high.append(high_idx)
            simplexs[rank+1][high_idx].low.append(cnts[rank])
    nodes_to_index[rank][tuple(nodes_new)] = cnts[rank]
    cnts[rank] += 1
    return graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, False, False



def Graph_reduce(graph:nx.Graph, max_rank, simplexs, nodes_to_index):
    cnts = {}
    # 记录一下少构建了多少？
    less_cnts = {}
    reduced_edge_list = []
    for i in range(max_rank+1):
        simplexs[i] = {}
        nodes_to_index[i] = {}
        cnts[i] = 0
        less_cnts[i] = 0
    # print(nx.is_frozen(graph))
    node_list = list(graph.nodes())
    node_list.sort()
    # print(nx.is_frozen(graph))
    for i in node_list:
        graph, simplexs, nodes_to_index, cnts, reduced_edge_list, less_cnts, vanish, high_vanish = offline_build(graph, max_rank, simplexs, nodes_to_index, 0, [], list(graph[i].keys()), i, 100, cnts, reduced_edge_list, less_cnts)
        # neighbors = list(graph[i].keys())
        # lenth = len(neighbors)
        # for nei in neighbors:
        #     if nei < i:
        #         continue
        #     graph, simplexs, nodes_to_index, cnts, reduced_edge_list, vanish, high_vanish = offline_build(graph, max_rank, simplexs, nodes_to_index, 1, [i], neighbors, nei, lenth, cnts, reduced_edge_list)
    # 考虑需要返回哪些数据用于后续。
    f = open('simplex_num.txt', 'a')
    f.write('Less\n')
    for i in range(max_rank+1):
        f.write('%ld ' % (less_cnts[i]))
    f.write('\nRest\n')
    for i in range(max_rank+1):
        f.write('%ld ' % (cnts[i]))
    f.write('\n\n')
    f.close()
    return graph, 1, simplexs, nodes_to_index, reduced_edge_list
            

# 将某一个单纯形的所有 coface 全部删除的同时进行 vanish
def clear_simplex(simplexs, sim:simplex, idx, rank, graph, reduced_edge_list):
    pid = os.getpid()
    global maximun_memory
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory = memory_info.rss
    if memory-current_memory > maximun_memory:
        maximun_memory = memory-current_memory
        print('Memory cost:', memory-current_memory)
    # 优先将所有预定要 vanish 的单纯形打上标签，并且将指向它的指针删除
    simplexs[rank][idx].vanished = True
    # 如果是边就对 graph 进行修改
    if rank == 1:
        graph.remove_edge(simplexs[1][idx].nodes[0], simplexs[1][idx].nodes[1])
        reduced_edge_list.append((simplexs[1][idx].nodes[0], simplexs[1][idx].nodes[1]))
    # 删除其刻面指向其的指针
    for other_coface in sim.low:
        # print(rank, idx)
        # print(simplexs[rank-1][other_coface].high)
        # print('fa', rank, idx, sim.high, sim.nodes, sim.low)
        # another = simplexs[rank-1][other_coface]
        # print('son', rank-1, other_coface, another.high, another.nodes, another.low)
        pos = simplexs[rank-1][other_coface].high.index(idx)
        simplexs[rank-1][other_coface].high = simplexs[rank-1][other_coface].high[:pos] + simplexs[rank-1][other_coface].high[pos+1:]
    # 高维单纯形继续 vanish
    for coface in sim.high:
        if (not simplexs[rank+1][coface].vanished):
            simplexs, graph, reduced_edge_list  = clear_simplex(simplexs, simplexs[rank+1][coface], coface, rank+1, graph, reduced_edge_list)

    if rank >= 2:
    # 对新的部分进行 vanish 
    # 遍历每一个刻面 （facet）
        for other_coface in sim.low:
            if simplexs[rank-1][other_coface].vanished:
                continue
            # 该单纯形可作为一个新的 collapse 中的 free face (最多只能删除边)
            if rank-1 >= 1 and len(simplexs[rank-1][other_coface].high) == 1:
                vanish_fa = simplexs[rank-1][other_coface].high[0]
                if (not simplexs[rank][vanish_fa].vanished) and len(simplexs[rank][vanish_fa].high) == 0:
                    # print("Situation 1")
                    simplexs, graph, reduced_edge_list = clear_simplex(simplexs, simplexs[rank-1][other_coface], other_coface, rank-1, graph, reduced_edge_list)
            # 该单纯形可作为一个新的 collapse 中的 Maximum face (最多只能删除边)
            elif rank-1 > 1 and len(simplexs[rank-1][other_coface].high) == 0:
                for facet in simplexs[rank-1][other_coface].low:
                    if  (not simplexs[rank-2][facet].vanished) and len(simplexs[rank-2][facet].high) == 1:
                        # print("Situation 2")
                        simplexs, graph, reduced_edge_list = clear_simplex(simplexs, simplexs[rank-2][facet], facet, rank-2, graph, reduced_edge_list)

        
    return simplexs, graph, reduced_edge_list


def reduce(graph:nx.Graph, rank, simplexs, nodes_to_index, point1, point2):
    simplexs = clear_simplex(simplexs, simplexs[1][nodes_to_index[1][(point1, point2)]], nodes_to_index[1][(point1, point2)], 1)
    reduced_edge_list = []
    for i in range(rank, 1, -1):
        # print('Reducing:', i)
        if i not in simplexs.keys():
            continue
        for idx, simp in simplexs[i].items():
            # if i == 2 and idx == 34:
            #     print(simplexs[i][idx])
            # print(i, idx, simplexs[i][idx])
            if simplexs[i][idx].vanished == True:
                continue
            for coface in simp.low:
                if simplexs[i-1][coface].vanished == True:
                    continue
                if len(simplexs[i-1][coface].high) == 1:
                    # 清除所有 i-1 维单纯形对该 i 维单纯形的记录 （即删除 i 维单纯形）
                    for other_coface in simp.low:
                        pos = simplexs[i-1][other_coface].high.index(idx)
                        simplexs[i-1][other_coface].high = simplexs[i-1][other_coface].high[:pos] + simplexs[i-1][other_coface].high[pos+1:]
                    # 清除所有 i-2 维单纯形对该 i-1 维单纯形的记录 （即删除 i-1 维单纯形）
                    for co_coface in simplexs[i-1][coface].low:
                        pos = simplexs[i-2][co_coface].high.index(coface)
                        simplexs[i-2][co_coface].high = simplexs[i-2][co_coface].high[:pos] + simplexs[i-2][co_coface].high[pos+1:]
                    # 如果 i-1 维单纯形是个边，那么同时删除它在 graph 上的记录
                    if i == 2:
                        graph.remove_edge(simplexs[i-1][coface].nodes[0], simplexs[i-1][coface].nodes[1])
                        reduced_edge_list.append((simplexs[i-1][coface].nodes[0], simplexs[i-1][coface].nodes[1]))
                    simplexs[i-1][coface].vanished = True
                    simplexs[i][idx].vanished = True
                    break
    return graph, simplexs, nodes_to_index, reduced_edge_list


def Modify(data, percentage, device, max_rank, reduce_edge, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, num_cluster, finished, node_to_cluster, max_cluster_size, old_graph):
    target_n = len(data.x) * percentage
    start = time.time_ns()
    while n_vanished < target_n:
        # print('edge_vanishing_cnt', edge_vanishing_cnt)
        # for i in simplexs[0].keys():
        #     print(i, len(simplexs[0][i].keys()))
        #     for idx in simplexs[0][i].keys():
        #         print(simplexs[0][i][idx].nodes, simplexs[0][i][idx].vanished)
        # for key in n:
        #     print('node', key.index, key.edgenode)
        if sum(finished) == num_cluster:    #重新将图进行分块
            print("!!!!")
            print('before')
            for cluster in range(0, num_cluster):
                print(graphs[cluster])
            # time.sleep(1)
            old_to_new = {}
            for cluster in range(0, num_cluster):
                for node in graphs[cluster].nodes():
                    old_to_new[node] = node
                    for vanished in n[node].nodes:
                        old_to_new[vanished] = node
            graph = nx.Graph()
            for g in graphs:
                graph.add_nodes_from(g.nodes)
                graph.add_edges_from(g.edges)
            
            # print(graph.nodes())
            # print(graph.edges())
            # print(node_to_cluster)
            
            for old_graph_edge in old_graph.edges():
                node1 = int(old_graph_edge[0])
                node2 = int(old_graph_edge[1])
                if node_to_cluster[node1] == node_to_cluster[node2]:
                    continue
                graph.add_edge(old_to_new[node1], old_to_new[node2])

            random_reduce = 0
            back_edges = []
            for edge in graph.edges():
                if (not n[edge[0]].train_node) or (not n[edge[1]].train_node):
                    random_reduce = 1
            if random_reduce == 0:
                while n_vanished < target_n:
                    for i in range(num_cluster):
                        target = n[random.choice(list(graphs[i].nodes()))] # 改为图中的任意一个点
                        target_neighbor = random.choice(list(graphs[i].nodes()))
                        # print(type(target.index), type(target_neighbor))
                        id = target.index
                        while target_neighbor == id or (n[target.index].train_node and n[target_neighbor].train_node):
                            target_neighbor = random.choice(list(graphs[i].nodes()))
                        
                        n[target_neighbor].nodes = n[target_neighbor].nodes + target.nodes + [target.index]
                        n[target_neighbor].ed_van = n[target_neighbor].ed_van + target.ed_van
                        n[target_neighbor].vanished += target.vanished
                        n[target_neighbor].train_node = (n[target.index].train_node or n[target_neighbor].train_node)
                        graphs[i].remove_node(target.index)
                        #print(neighbor[0], target.index)
                        n_vanished += 1
                        node_vanished[target.index] = 1
                continue
            # print(graph.nodes())
            # print(graph.edges())
            num_nodes = data.x.size()[0]
            new_node_to_cluster = list(0 for i in range(0, num_nodes))
            new_graphs = []
            cnt_cluster_node = 0
            current_nodes = []
            i = 1
            # print(max_cluster_size, cnt_cluster_node)
            
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
                            i += 1
                            new_graphs.append(graph.subgraph(current_nodes))
                            cluster_node.append(current_nodes)
                            current_nodes = []
                            while not q1.empty():
                                q1.get()
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
                        new_graphs.append(graph.subgraph(current_nodes))
                        cluster_node.append(current_nodes)
                        i += 1
                        cnt_cluster_node = 0
                        current_nodes = []
            if len(current_nodes) != 0:
                new_graphs.append(graph.subgraph(current_nodes))
                cluster_node.append(current_nodes)
            global maximun_memory
            pid = os.getpid()

            process = psutil.Process(pid)
            memory_info = process.memory_info()
            memory = memory_info.rss
            if memory-current_memory > maximun_memory:
                maximun_memory = memory-current_memory
                print('3 Memory cost:', memory-current_memory)
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
            old_graph = graph
            # print(num_cluster, len(graphs), len(simplexs), len(nodes_to_index))
            for i in range(num_cluster):
                # print(i)
                # print(graphs[i])
                # print(simplexs[i])
                # print(nodes_to_index[i])
                graphs[i], flag_none[i], simplexs[i], nodes_to_index[i], reduced_edge_list = Graph_reduce(graphs[i], 1, simplexs[i], nodes_to_index[i])
            
            for i in range(num_cluster):
                # print(graphs[i])
                # print(graphs[i].edges())
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
            print('after')
            # print(len(graphs))
            for cluster in range(0, num_cluster):
                print(graphs[cluster])
        for cluster in range(0, num_cluster):
            # print('here')
            # print(n[2].index, n[2].train_node)
            if n_vanished % 500 == 0:
                print('n_vanished', n_vanished)
            if finished[cluster] == 1:
                continue
            if n_vanished >= target_n:
                break
            graph = nx.Graph(graphs[cluster])
            if len(list(graph.nodes())) <= max_cluster_size / 2 and len(graphs) > 1:
                finished[cluster] = 1
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
                print("all %d nodes are vanished" % n_vanished)
                finished[cluster] = 1
                break
            # q[cluster].join()
            target = q[cluster].get()
            
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
            neighbor = list(graph[target.index].keys())
            # print('target', simplexs[cluster][0][nodes_to_index[cluster][0][target.index]].nodes)
            # print('neighbor', neighbor)
            # print('q', q)
            # print(n[2].index, n[2].train_node)
            # if len(list(graph.nodes())) == 522:
            #     print(target.index, neighbor)
            if len(neighbor) == 0:                  # 可能出现无法继续消除的情况，到极限就退出
                # print("??!")
                continue
            elif len(neighbor) > 1:
                if not reduce_edge:
                    print("Can't go forward.")
                    break
                if (not flag_none[cluster]) and reduced_rank[cluster] < max_rank:
                    reduced_rank[cluster] += 1
                    graph, flag_none[cluster], simplexs[cluster], nodes_to_index[cluster], reduced_edge_list = Graph_reduce(graph, reduced_rank[cluster], simplexs[cluster], nodes_to_index[cluster])
                    # print('%d list_len: ' % reduced_rank[cluster], len(reduced_edge_list))
                    for i, j in reduced_edge_list:
                        n[i].edgenode -= 1
                        n[i].vanished += 1
                        n[j].edgenode -= 1
                        # print('putting', n[i].index, n[i].edgenode, n[j].index, n[j].edgenode)
                        q[cluster].put(copy.deepcopy(n[i]))
                        q[cluster].put(copy.deepcopy(n[j]))
                    # print('putting', n[target.index].index, n[target.index].edgenode)
                    q[cluster].put(copy.deepcopy(n[target.index]))
                    graphs[cluster] = graph
                    continue
                if edge_vanishing_cnt % 50 == 0:
                    print('reduce edge: ' , edge_vanishing_cnt)
                edge_vanishing_cnt += 1
                # print('Do edge vanishing:', edge_vanishing_cnt)

                q[cluster].put(copy.deepcopy(n[target.index]))
                target_edge = random.choice(list(graphs[cluster].edges()))
                target = n[target_edge[0]]
                node_2 = target_edge[1]


                # node_2 = neighbor[-1]
                # graph.remove_edge(target.index, node_2)
                
                n1 = min(target.index, node_2)
                n2 = max(target.index, node_2)
                if simplexs[cluster][1][nodes_to_index[cluster][1][(n1, n2)]].vanished:
                    print('something wrong here')
                reduced_edge_list = []
                simplexs[cluster], graph, reduced_edge_list = clear_simplex(simplexs[cluster], simplexs[cluster][1][nodes_to_index[cluster][1][(n1, n2)]], nodes_to_index[cluster][1][(n1, n2)], 1, graph, reduced_edge_list)
                # graph, simplexs[cluster], nodes_to_index[cluster], reduced_edge_list = reduce(graph, , simplexs[cluster], nodes_to_index[cluster])
                
                for i, j in reduced_edge_list:
                    n[i].edgenode -= 1
                    n[i].vanished += 1
                    n[j].edgenode -= 1
                    # print('putting', n[i].index, n[i].edgenode, n[j].index, n[j].edgenode)
                    q[cluster].put(copy.deepcopy(n[i]))
                    q[cluster].put(copy.deepcopy(n[j]))
                if len(reduced_edge_list) > 1:
                    print("Extra: %d edges." % len(reduced_edge_list))

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
                graphs[cluster] = graph
                continue
            # print(n[neighbor[0]].index, n[neighbor[0]].train_node)
            # print(n[2].index, n[2].train_node)
            # print(target.index, target.train_node)
            if n[neighbor[0]].train_node and n[target.index].train_node:
                # print('two trian')
                continue
            else:
                n[neighbor[0]].nodes = n[neighbor[0]].nodes + target.nodes + [target.index]
                n[neighbor[0]].ed_van = n[neighbor[0]].ed_van + target.ed_van
                n[target.index].edgenode -= 1
                n[neighbor[0]].edgenode -= 1
                n[neighbor[0]].train_node = (n[neighbor[0]].train_node or n[target.index].train_node)
                n[neighbor[0]].vanished += target.vanished
                # print('putting', n[neighbor[0]].index, n[neighbor[0]].edgenode)
                q[cluster].put(copy.deepcopy(n[neighbor[0]]))
                # print('removing', target.index)
                # print(target.index, neighbor, list(graph[target.index].keys()))
                # print(target.index, simplexs[cluster][0][nodes_to_index[cluster][0][tuple([target.index])]].nodes)
                simplexs[cluster], graph, reduced_edge_list = clear_simplex(simplexs[cluster], simplexs[cluster][0][nodes_to_index[cluster][0][tuple([target.index])]], nodes_to_index[cluster][0][tuple([target.index])], 0, graph, [])
                # print(reduced_edge_list)
                graph.remove_node(target.index)
                n_vanished += 1
                node_vanished[target.index] = 1
                graphs[cluster] = graph
    end = time.time_ns()
    f = open('time_record_1000', 'a')
    f.write('Cora, %ld\n' % ((end-start)//1000000))
    f.close()
    # res = copy.deepcopy(data).to(device)
    # res.new_x = copy.deepcopy(res.x)
    # flag = 0
    # old_to_new = {}
    # new_to_old = {}
    # for cluster in range(0, num_cluster):
    #     # print(graph[cluster])
    #     # print(graphs[cluster].nodes())
    #     for node in graphs[cluster].nodes():
    #         # print(node)
    #         if flag == 0: # 记录当前点数量
    #             res.new_x = copy.deepcopy(res.x[node].unsqueeze(0))
    #             old_to_new[node] = flag
    #             new_to_old[flag] = node
    #             flag += 1
    #         else:
    #             res.new_x = torch.cat((res.new_x, res.x[node].unsqueeze(0)), 0)
    #             old_to_new[node] = flag
    #             new_to_old[flag] = node
    #             flag += 1
    # # 对于图上的每一个点
    # for cluster in range(0, num_cluster):
    #     for node in graphs[cluster].nodes():
    #         # sub 为当前保留下的节点的特征
    #         sub = res.x[node].to(device)
    #         # 将 vanished 中的所有的节点特征相加，同时将其指向当前的点
    #         for vanished in n[node].nodes:
    #             old_to_new[vanished] = old_to_new[node]
    #             sub = sub + res.x[vanished]
    #         # # 将由于去边所获得的特征加入
    #         # for vanished in n[node].ed_van:
    #         #     sub = sub + res.x[vanished]
    #         # 取其中均值
    #         if len(n[node].nodes) != 0 or len(n[node].ed_van) != 0:
    #             res.new_x[old_to_new[node]] = sub / float((1+len(n[node].nodes)+len(n[node].ed_van)))
    # l1 = []
    # l2 = []
    # # print(old_to_new)
    # edges = {}
    # cnt = 0
    # # 将图之间的边加上
    # # print(old_to_new[64])
    # for edge_num in range(data.edge_index.size()[1]):
    #     # print(old_to_new[int(data.edge_index[0][edge_num])])
    #     # print('here', old_to_new[3])
    #     # print(data.edge_index[0][edge_num])
    #     # print(old_to_new[int(data.edge_index[0][edge_num])])
    #     # print(data.edge_index[1][edge_num])
    #     # print(old_to_new[int(data.edge_index[1][edge_num])])
    #     node1 = old_to_new[int(data.edge_index[0][edge_num])] 
    #     node2 = old_to_new[int(data.edge_index[1][edge_num])]
    #     if node_to_cluster[int(data.edge_index[0][edge_num])] == node_to_cluster[int(data.edge_index[1][edge_num])]:
    #         cnt += 1
    #         continue
    #     if (min(node1, node2), max(node1, node2)) in edges.keys():
    #         # print("exist edge")
    #         continue
    #     l1.append(node1)
    #     l2.append(node2)
    #     l1.append(node2)  
    #     l2.append(node1)
    #     edges[(min(node1, node2), max(node1, node2))] = 1
    #     # 如果是arxiv的话需要取消注释
    # # print(cnt)
    # for cluster in range(0, num_cluster):
    #     for edge in graphs[cluster].edges():
    #         l1.append(old_to_new[edge[0]])
    #         l2.append(old_to_new[edge[1]])
    #         l1.append(old_to_new[edge[1]])
    #         l2.append(old_to_new[edge[0]])
    # res.new_edge = torch.cat((torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0)), 0)
    # print(old_to_new)
    
    # global maximun_memory
    print(maximun_memory)
    return res, old_to_new, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


dataname = 'Citeseer'
if dataname == 'dblp':
    dataset = CitationFull(root='./dataset', name=dataname)
elif dataname == 'Physics':
    dataset = Coauthor(root='./dataset', name=dataname)
elif dataname == 'OGBN-arxiv':
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = './')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator('ogbn-arxiv')
elif dataname == 'pubmed':
    dataset = Planetoid(root='./dataset', name=dataname)
else:
    dataset = Planetoid(root='./', name=dataname)

# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = './')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
data = dataset[0].to(device)
if dataname == 'OGBN-arxiv':
    data.y = torch.squeeze(data.y)
print(data)
if dataname == 'dblp' or dataname == 'Physics' or dataname == 'OGBN-arxiv':
    indices = []
    num_classes = torch.unique(data.y,return_counts=True)[0].shape[0]
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    test_index = torch.cat([i[50:] for i in indices], dim=0)

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
print(torch.unique(data.y,return_counts=True))
print(torch.unique(data.y[data.train_mask],return_counts=True))

# data.x = torch.zeros([6, 1433])
# data.edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4], [4, 5], [1, 0]]).t()


import psutil
import os
import time

# 获取当前进程ID
pid = os.getpid()

# 创建一个记录内存使用的列表
memory_usage = []

process = psutil.Process(pid)
memory_info = process.memory_info()
current_memory = memory_info.rss
print(current_memory)
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
for max_cluster_size in [1000]:
    n = []
    graph = nx.Graph()
    for i in range(len(data.x)):
        graph.add_node(i)
        n.append(nodes(i))
        if data.train_mask[i] == 1:
            n[i].train_node = True
        # if data.val_mask[i] == 1:
        #     n[i].train_node = True
            # print('train_node', i)
    # n[2].train_node = False
    # n[3].train_node = False
    # n[0].train_node = False
    # n[5].train_node = False
    # print(graph.nodes())
    for i in range(len(data.edge_index[0])): 
        graph.add_edge(int(data.edge_index[0, i]), int(data.edge_index[1, i]))
        # if int(data.edge_index[0, i]) == 0 or int(data.edge_index[1, i]) == 0:
        #     print("???")
        # n[int(data.edge_index[0, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边
        # n[int(data.edge_index[1, i])].edgenode += 1              # 注意在arxiv数据集中要改为双个端点都加边

    cnt_cluster_node = 0
    num_nodes = data.x.size()[0]
    node_to_cluster = list(0 for i in range(0, num_nodes))
    graphs = []
    cluster_node = []
    num_cluster = (num_nodes-1) // max_cluster_size + 1 
    print(num_cluster)
    for i in range(1, num_cluster):
        cnt_cluster_node = 0
        current_nodes = []
        # print(max_cluster_size, cnt_cluster_node)
        while max_cluster_size > cnt_cluster_node:
            q1 = Queue()
            for node in range(num_nodes):
                if not node_to_cluster[node]:
                    q1.put(copy.deepcopy(node))
                    break
            while not q1.empty():
                node = q1.get()
                if node_to_cluster[node] != 0:
                    continue
                current_nodes.append(node)
                node_to_cluster[node] = i
                # print(current_nodes)
                cnt_cluster_node += 1
                if max_cluster_size <= cnt_cluster_node:
                    break
                neighbor = list(graph[node].keys())
                for node2 in neighbor:
                    if not node_to_cluster[node2]:
                        q1.put(copy.deepcopy(node2))
        # print(current_nodes)
        graphs.append(nx.Graph(graph.subgraph(current_nodes)))
        cluster_node.append(current_nodes)
    current_nodes = []
    for node in range(num_nodes):
        if not node_to_cluster[node]:
            current_nodes.append(node)
    graphs.append(nx.Graph(graph.subgraph(current_nodes)))
    print(nx.is_frozen(graphs[0]))
    print(nx.is_frozen(graphs[1]))
    # print(nx.is_frozen(graphs[0]))
    cluster_node.append(current_nodes)
    # for i in range(len(graphs)):
    #     print(graphs[i])
    flag_none = list(0 for i in range(0, num_cluster))
    # 将节点映射到单纯性
    nodes_to_index = list({} for i in range(0, num_cluster))
    reduced_rank = list(6 for i in range(0, num_cluster))
    simplexs = list({} for i in range(0, num_cluster))
    for i in range(num_cluster):
        graphs[i], flag_none[i], simplexs[i], nodes_to_index[i], reduced_edge_list = Graph_reduce(graphs[i], 6, simplexs[i], nodes_to_index[i])
        for i, j in reduced_edge_list:
            n[i].edgenode -= 1
            n[i].vanished += 1
            n[j].edgenode -= 1
    edge_vanishing_cnt = 0
    cluster_n_nanished = list(0 for i in range(0, num_cluster))
    q = list(PriorityQueue() for i in range(0, num_cluster))
    finished = list(0 for i in range(0, num_cluster))
    n_vanished = 0

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
    #     print(nx.is_frozen(graphs[i]))
    node_vanished = {}  # 已经被删除的点
    # q = PriorityQueue()
    # for i in n:
    #     q.put(i)

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


    # for v in np.arange(0.00, 1, 0.05):
    for v in [0.9]:
        print(type(data))
        
        try:
            file = open('Reduce.txt', mode='a', encoding='utf-8')
        except:
            print(f'open analyse file error!')
        file.write('vanished: %lf' % v)
        file.write('\n')
        file.close()
        data_mol, m, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, finished, num_cluster, node_to_cluster = Modify(data, v, device, 6, True, n, graphs, flag_none, simplexs, nodes_to_index, edge_vanishing_cnt, n_vanished, node_vanished, q, reduced_rank, num_cluster, finished, node_to_cluster, max_cluster_size, graph)
        data_mol = data_mol.to(device)
        # train_index = torch.zeros(data_mol.new_x.shape[0])

        # for i in range(int(data_mol.train_index.shape[0])):
        #     if m[i] = 1
        
        np.save('./Reduced_Node_Data/%s_%.2f_split%d_offline.npy' % (dataname, v, max_cluster_size), (data_mol.cpu(), m))
        n_cnt = {}
        for i in range(data_mol.new_edge.size()[1]):
            n_cnt[int(data_mol.new_edge[0][i])] = 1
            n_cnt[int(data_mol.new_edge[1][i])] = 1
        print('len', len(n_cnt.keys()))

        # print('edge_vanishing_cnt', edge_vanishing_cnt)
        # for i in simplexs[0].keys():
        #     print(i, len(simplexs[0][i].keys()))
        #     for idx in simplexs[0][i].keys():
        #         print(simplexs[0][i][idx].nodes, simplexs[0][i][idx].vanished)
        print(data_mol)
        # print(data_mol.new_edge)
    