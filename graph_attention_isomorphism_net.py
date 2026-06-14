from typing import Callable, Optional, Union

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, ReLU, Identity

from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, PairTensor

class MLP(torch.nn.Module):
    """Two-layer MLP with batch normalization and ReLU after each layer.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.

    Inputs:
        x (Tensor): Input (N, in_channels).

    Returns:
        Tensor: Output (N, out_channels).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear1 = Linear(in_channels, out_channels, weight_initializer='kaiming_uniform')
        self.bn1 = BatchNorm1d(out_channels)
        self.a1 = ReLU(True)
        
        self.linear2 = Linear(out_channels, out_channels, weight_initializer='kaiming_uniform')
        self.bn2 = BatchNorm1d(out_channels)
        self.a2 = ReLU(True)
    
    def reset_parameters(self):
        """Reset linear layer parameters.

        Args:
            None.

        Returns:
            None.
        """
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        
    
    def forward(self, x):
        """Apply two linear-bn-relu blocks.

        Args:
            x (Tensor): Input (N, in_channels).

        Returns:
            Tensor: Output (N, out_channels).
        """
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.a1(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.a2(x)
        
        return x 
        
        
class SelfAttentionBlock(torch.nn.Module):
    """Graph attention followed by MLP output with residual skip connection.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        key_query_len (int, optional): Internal dimension for attention. Defaults to out_channels.

    Inputs:
        x (Tensor): Node features (N, in_channels).
        edge_index (Tensor): Edge indices (2, E).

    Returns:
        Tensor: Output features (N, out_channels).
    """
    def __init__(self, in_channels, out_channels, key_query_len=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(key_query_len, int) & key_query_len > 0:
            self.key_query_len = key_query_len
        else:
            self.key_query_len = out_channels
                
        self.graph_attention = GATConv(in_channels=in_channels, out_channels=out_channels)
        
        self.mlp_out = MLP(self.key_query_len, out_channels)
        
        if out_channels == in_channels:
            self.lin_skip = Identity()
        else:
            self.lin_skip = Linear(in_channels, out_channels, weight_initializer='kaiming_uniform')
        
        self.reset_parameters()
     
    def reset_parameters(self):
        """Reset MLP parameters.

        Args:
            None.

        Returns:
            None.
        """
        self.mlp_out.reset_parameters()
    
    def forward(self, x, edge_index):
        """Apply GAT, then MLP, add skip connection.

        Args:
            x (Tensor): Node features (N, in_channels).
            edge_index (Tensor): Edge indices (2, E).

        Returns:
            Tensor: Output (N, out_channels).
        """
        out = self.graph_attention(x, edge_index)
        out = self.mlp_out(out) + self.lin_skip(x)
        
        return out
    

class GAINet(MessagePassing):
    """Graph Attention Isomorphism Network (GAINet).

    Propagates neighbor messages via sum aggregation, applies self-attention on
    both the neighbor and skip branches, combines them, and applies output attention.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        key_query_len (int, optional): Internal attention dimension. Defaults to out_channels.
        aggr (str): Aggregation method. Default: 'sum'.
        **kwargs: Additional MessagePassing arguments.

    Inputs:
        x (Tensor): Node features (N, in_channels).
        edge_index (Tensor): Edge indices (2, E).

    Returns:
        Tensor: Output node features (N, out_channels).
    """
    def __init__(self,  
                in_channels: int, 
                out_channels: int,
                key_query_len: int = None, 
                aggr: str = 'sum', 
                **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(key_query_len, int) & key_query_len > 0:
            self.key_query_len = key_query_len
        else:
            self.key_query_len = out_channels
            
        self.self_attention_neighbor = SelfAttentionBlock(in_channels=in_channels,
                                                          out_channels=out_channels,
                                                          key_query_len=key_query_len)

        self.self_attention_skip = SelfAttentionBlock(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        key_query_len=key_query_len)
        
        self.self_attention_out = SelfAttentionBlock(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    key_query_len=key_query_len)
        
        self.lin_epsilon = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
    
    def reset_parameters(self):
        """Reset all self-attention blocks.

        Args:
            None.

        Returns:
            None.
        """
        self.self_attention_neighbor.reset_parameters()
        self.self_attention_skip.reset_parameters()
        self.self_attention_out.reset_parameters()
        
        
    def forward(self, x: Tensor, edge_index: Adj):
        """Aggregate neighbors, apply self-attention on neighbor and skip branches, combine and apply output attention.

        Args:
            x (Tensor): Node features (N, in_channels).
            edge_index (Tensor): Edge indices (2, E).

        Returns:
            Tensor: Output features (N, out_channels).
        """
        neighbor = self.propagate(edge_index, x=x, size=None)
        neighbor = self.self_attention_neighbor(neighbor, edge_index)
        self_skip = (1+self.lin_epsilon)*x
        self_skip = self.self_attention_skip(self_skip, edge_index)
        out = self_skip + neighbor
        out = self.self_attention_out(out, edge_index)
        
        return out
        
        
    def message(self, x_j: Tensor):
        """Pass through neighbor features.

        Args:
            x_j (Tensor): Neighbor features (E, in_channels).

        Returns:
            Tensor: Unmodified neighbor features (E, in_channels).
        """
        return x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
