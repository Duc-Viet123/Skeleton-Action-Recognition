import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.utils.tgcn import ConvTemporalGraphical
    from model.utils.graph import Graph
except ModuleNotFoundError:
    from .utils.tgcn import ConvTemporalGraphical
    from .utils.graph import Graph


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        num_class,
        graph_args,
        edge_importance_weighting,
        **kwargs,
    ):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        kernel_size = (9, A.size(0))
        dropout = kwargs.get("dropout", 0)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(in_channels, 64, kernel_size, stride=1, residual=False, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(64, 128, kernel_size, stride=2, dropout=dropout),
                STGCNBlock(128, 128, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(128, 128, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(128, 256, kernel_size, stride=2, dropout=dropout),
                STGCNBlock(256, 256, kernel_size, stride=1, dropout=dropout),
                STGCNBlock(256, 256, kernel_size, stride=1, dropout=dropout),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, **kwargs):
        N, C, T, V, M = x.size()
        x = self._normalize_input(x, N, C, T, V, M)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        x = self._normalize_input(x, N, C, T, V, M)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        feature_map = x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1), feature_map

    def _normalize_input(self, x, N, C, T, V, M):
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x.view(N * M, C, T, V)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be (temporal_kernel_size, spatial_kernel_size)")
        if kernel_size[0] % 2 != 1:
            raise ValueError("temporal kernel size must be odd")

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
