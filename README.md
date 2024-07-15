# Topology-preserving-Graph-Coarsening
## Datasets

Because some datasets used in the paper are too large to be uploaded to GitHub, we have summarized the links for the dataset in the table below.

Datasets used for performance studies.

| Dataset | Link |
| --- | --- |
| pubmed | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid |
| DBLP | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CitationFull.html#torch_geometric.datasets.CitationFull |
| Coauthor Physics | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html#torch_geometric.datasets.Coauthor |
| Reddit | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html#torch_geometric.datasets.Reddit |
| OGBN-arxiv | https://ogb.stanford.edu/docs/nodeprop/#ogbn-products |
| OGBN-products | https://ogb.stanford.edu/docs/nodeprop/#ogbn-products |
| dblp-v7 | https://www.aminer.org/citation |
| dblp-v5 | https://www.aminer.org/citation |
| cit-Patents | https://snap.stanford.edu/data/cit-Patents.html |
| com-youtube | https://snap.stanford.edu/data/com-Youtube.html |


Datasets used for case studies are uploaded to the “Graph_Making” folder. 

## Preprocess

The datasets need to be preprocessed, since our paper only focuses on undirected graph. 


## Usage of Bottom-up GEC Algorithms

Datasets need to be stored in the “dataset” folder. Please remember to change the "dataname" and the path of dataset to run on different dataset. For different coarsening ratio, please change the and the "ratio_list" in the python file. To run the program of Bottom-up GEC, run the following command. 

```
python Bottom_up_all_simplex.py
```

The coarsened graphs will be stored in the “Reduced_Node_Data", the coarsend graph are saved in ".npy" format, each file contains a tuple (coarsened_dataset, mapping relationship).

To run the training program on GCN network, run the following command.

```
python train.py
```

