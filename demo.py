from torch_geometric.datasets import TUDataset, Planetoid
import torch_geometric.transforms as T
import os.path as osp

if __name__ == "__main__":
    ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "Planetoid")
    dataset = Planetoid(ROOT, "citeseer", transform=T.NormalizeFeatures())
    print(dataset)
