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
        
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.a1(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.a2(x)
        
        return x 
        
        
class SelfAttentionBlock(torch.nn.Module):
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
    
        self.mlp_out.reset_parameters()
    
    def forward(self, x, edge_index):
                
        out = self.graph_attention(x, edge_index)
        out = self.mlp_out(out) + self.lin_skip(x)
        
        return out
    

class GAINet(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
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
        self.self_attention_neighbor.reset_parameters()
        self.self_attention_skip.reset_parameters()
        self.self_attention_out.reset_parameters()
        
        
    def forward(self, x: Tensor, edge_index: Adj):
        
        neighbor = self.propagate(edge_index, x=x, size=None)
        neighbor = self.self_attention_neighbor(neighbor, edge_index)
        self_skip = (1+self.lin_epsilon)*x
        self_skip = self.self_attention_skip(self_skip, edge_index)
        out = self_skip + neighbor
        out = self.self_attention_out(out, edge_index)
        
        return out
        
        
    def message(self, x_j: Tensor):
        return x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
