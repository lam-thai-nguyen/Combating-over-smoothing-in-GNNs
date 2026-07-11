import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=128, num_layers=3, dropout=0.5):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
        for _ in range(num_layers-2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            
        logits = self.convs[-1](x, edge_index)
        return logits
    