import copy
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cognitive_graph import options
from cognitive_graph.datasets import build_dataset
from cognitive_graph.models import build_model

from . import BaseTask, register_task


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on

    def __init__(self, args):
        super(NodeClassification, self).__init__(args)

        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data.cuda()
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        model = build_model(args)
        self.model = model.cuda()
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self._compute_A()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        test_acc, _ = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        return dict(
            Acc=test_acc
        )

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        F.nll_loss(
            self.model(self.data.x, self.data.A)[self.data.train_mask],
            self.data.y[self.data.train_mask],
        ).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        logits = self.model(self.data.x, self.data.A)
        loss = F.nll_loss(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
        )
        _, mask = list(self.data(f"{split}_mask"))[0]
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss

    def _compute_A(self):
        edge_index = self.data.edge_index.cpu().numpy()
        adj = sp.csr_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1]))
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1)) ** (-0.5)
        D2 = np.array(adj.sum(axis=0)) ** (-0.5)
        D1 = sp.diags(D1[:, 0], format="csr")
        D2 = sp.diags(D2[0, :], format="csr")
        A = adj.dot(D1)
        self.data.A = D2.dot(A)

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
            )
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        self.data.A = sparse_mx_to_torch_sparse_tensor(self.data.A).cuda()
        adj = normalize(adj)
        self.data.adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
