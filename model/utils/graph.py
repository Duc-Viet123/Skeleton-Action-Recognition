import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    Dn = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)
    for i, value in enumerate(Dl):
        if value > 0:
            Dn[i, i] = value ** (-1)
    return np.dot(A, Dn)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """Skeleton graph builder used by ST-GCN."""

    def __init__(self, layout='openpose', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            # Matches Feeders.feeder_finetune.FeederMultiClass.bone_pairs.
            inward = [
                (0, 1), (2, 1), (3, 2), (4, 3),
                (5, 1), (6, 5), (7, 6),
                (8, 1), (9, 8), (10, 9), (11, 10),
                (12, 8), (13, 12), (14, 13),
                (15, 0), (16, 0), (17, 15),
            ]
            outward = [(j, i) for i, j in inward]
            self.edge = self_link + inward + outward
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12),
            ]
            inward = [(i - 1, j - 1) for i, j in inward_ori_index]
            outward = [(j, i) for i, j in inward]
            self.edge = self_link + inward + outward
            self.center = 21 - 1
        else:
            raise ValueError(f"Unsupported graph layout: {layout}")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            self.A = normalize_adjacency[None, :, :]
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node), dtype=np.float32)
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node), dtype=np.float32)
                a_close = np.zeros((self.num_node, self.num_node), dtype=np.float32)
                a_further = np.zeros((self.num_node, self.num_node), dtype=np.float32)
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError(f"Unsupported graph strategy: {strategy}")
