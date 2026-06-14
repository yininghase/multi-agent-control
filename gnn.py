import torch
from torch.nn import ReLU, Tanh, BatchNorm1d
from numpy import pi
from torch_geometric.nn import Linear, TransformerConv, EdgeConv
from itertools import permutations

from u_attention_conv import MyTransformerConv
from graph_attention_isomorphism_net import GAINet


class ConvResidualBlock(torch.nn.Module):
    """Residual block with two graph convolutions and batch normalization.

    Args:
        io_node_num (int): Input/output feature dimension.
        hidden_node_num (int): Hidden dimension for first convolution.
        key_query_len (int): Key/query dimension for attention-based convs. Default: 512.
        conv_type (str): Type of convolution: 'UAttentionConv', 'GAINet', 'TransformerConv', or 'EdgeConv'.

    Inputs:
        x0 (Tensor): Node features (N, io_node_num).
        edges (Tensor): Edge indices (2, E).

    Returns:
        Tensor: Output features (N, io_node_num).
    """
    def __init__(self, io_node_num, hidden_node_num, key_query_len=512, conv_type="UAttentionConv"):
        super().__init__()
        
        if conv_type == "UAttentionConv":
            self.conv1 = MyTransformerConv(io_node_num, hidden_node_num, key_query_len=key_query_len)
            self.bn1 = BatchNorm1d(hidden_node_num)
            self.a1 = ReLU()
            self.conv2 = MyTransformerConv(hidden_node_num, io_node_num, key_query_len=key_query_len)
            self.bn2 = BatchNorm1d(io_node_num)
            self.a2 = ReLU()
        
        elif conv_type == "GAINet":
            self.conv1 = GAINet(io_node_num, hidden_node_num, key_query_len=key_query_len)
            self.bn1 = BatchNorm1d(hidden_node_num)
            self.a1 = ReLU()
            self.conv2 = GAINet(hidden_node_num, io_node_num, key_query_len=key_query_len)
            self.bn2 = BatchNorm1d(io_node_num)
            self.a2 = ReLU()
        
        elif conv_type == "TransformerConv":
            
            self.conv1 = TransformerConv(io_node_num, hidden_node_num, heads=1, concat=False)
            self.bn1 = BatchNorm1d(hidden_node_num)
            self.a1 = ReLU()
            self.conv2 = TransformerConv(hidden_node_num, io_node_num, heads=1, concat=False)
            self.bn2 = BatchNorm1d(io_node_num)
            self.a2 = ReLU()
        
        elif conv_type == "EdgeConv":
            
            self.conv1 = EdgeConv(nn=Linear(2*io_node_num, hidden_node_num, weight_initializer='kaiming_uniform'))
            self.bn1 = BatchNorm1d(hidden_node_num)
            self.a1 = ReLU()
            self.conv2 = EdgeConv(nn=Linear(2*hidden_node_num, io_node_num, weight_initializer='kaiming_uniform'))
            self.bn2 = BatchNorm1d(io_node_num)
            self.a2 = ReLU()
        
        else:
            raise NotImplementedError("Not implement this type of GNN convolution!")
        
    def forward(self, x0, edges):
        """Apply two convolutions with residual connection.

        Args:
            x0 (Tensor): Input node features (N, io_node_num).
            edges (Tensor): Edge indices (2, E).

        Returns:
            Tensor: Output node features (N, io_node_num).
        """
        x = self.conv1(x0, edges)
        x = self.bn1(x)
        x = self.a1(x)
        
        x = self.conv2(x, edges)
        x = self.bn2(x)
        x = x+x0
        x = self.a2(x)
                
        return x


class LinearBlock(torch.nn.Module):
    """Linear layer with batch normalization and configurable activation.

    Args:
        in_node_num (int): Input dimension.
        out_node_num (int): Output dimension.
        activation (str): Activation type, 'relu' or 'tanh'. Default: 'relu'.

    Inputs:
        x (Tensor): Input features (N, in_node_num).
        x0 (Tensor, optional): Residual connection tensor (N, out_node_num).

    Returns:
        Tensor: Output features (N, out_node_num).
    """
    def __init__(self, in_node_num, out_node_num, activation="relu"):
        super().__init__()
        
        self.linear = Linear(in_node_num, out_node_num, weight_initializer='kaiming_uniform')
        self.bn = BatchNorm1d(out_node_num)
        
        if activation == "relu":
            self.a = ReLU()
        elif activation == "tanh":
            self.a = Tanh()
        else:
            raise NotImplementedError("Not implement this type of activation function!")

        
    def forward(self, x, x0=None):
        """Apply linear transform, batch norm, optional residual, and activation.

        Args:
            x (Tensor): Input features (N, in_node_num).
            x0 (Tensor, optional): Residual to add before activation (N, out_node_num).

        Returns:
            Tensor: Output features (N, out_node_num).
        """
        x = self.linear(x)
        x = self.bn(x)
        
        if x0 is not None:
            x=x+x0
        
        x = self.a(x)
        
        return x


class LinearResidualBlock(torch.nn.Module):
    """Two-layer MLP with residual connection.

    Args:
        io_node_num (int): Input and output dimension.
        hidden_node_num (int): Hidden dimension.

    Inputs:
        x0 (Tensor): Input features (N, io_node_num).

    Returns:
        Tensor: Output features (N, io_node_num).
    """
    def __init__(self, io_node_num, hidden_node_num):
        super().__init__()
        
        self.linear1 = LinearBlock(io_node_num, hidden_node_num)
        self.linear2 = LinearBlock(hidden_node_num, io_node_num)
    
        
    def forward(self, x0):
        """Apply two linear blocks with residual connection.

        Args:
            x0 (Tensor): Input (N, io_node_num).

        Returns:
            Tensor: Output (N, io_node_num).
        """
        x = self.linear1(x0)
        x = self.linear2(x,x0)
        
        return x
       

class IterativeGNNModel(torch.nn.Module):
    """Iterative GNN model for multi-agent control: encodes state, applies graph convolution blocks, decodes controls.

    In training mode, outputs controls for a single step. In inference mode, auto-regressively
    rolls out a trajectory over the given horizon using a bicycle kinematics model.

    Args:
        horizon (int): Prediction horizon length.
        max_num_vehicles (int): Maximum number of vehicles supported.
        max_num_obstacles (int): Maximum number of obstacles supported.
        device (str): Device to use ('cpu' or 'cuda'). Default: 'cpu'.
        mode (str): 'training' or 'inference'. Default: 'inference'.
        conv_type (str): GNN convolution type. Default: 'UAttentionConv'.

    Inputs (forward):
        x0 (Tensor): Initial state features (total_nodes, 8).
        batches (Tensor): Problem configuration matrix (B, 2) with columns [num_nodes, num_vehicles].

    Returns:
        If mode='training': list of [vehicle_controls, static_controls].
        If mode='inference': tuple (controls, statics, states, edges_vehicles, edges_obstacles).
    """
    def __init__(self, horizon, max_num_vehicles, max_num_obstacles, device='cpu',
                 mode="inference", conv_type="UAttentionConv"):
        super().__init__()
        self.device = device
        self.horizon = horizon
        self.dt = 0.2
        self.input_length = 8
        self.output_length = 2
        self.bound = torch.tensor([1, 0.8]).to(self.device)
        self.mode = mode
        self.conv_type = conv_type
        self.no_vehicle_edges = False
        
        if conv_type.split('_')[-1] == "NoVehicleEdges":
            self.conv_type = conv_type[:-15]
            self.no_vehicle_edges = True
            
        self.max_num_vehicles = max_num_vehicles
        self.max_num_obstacles = max_num_obstacles
        
        self.edge_template = self.generate_edge_template()
        
        self.block0 = LinearBlock(self.input_length,80)
        self.block1 = ConvResidualBlock(80,160,conv_type=self.conv_type)
        self.block2 = ConvResidualBlock(80,160,conv_type=self.conv_type)
        self.block3 = LinearBlock(80, self.output_length, activation="tanh")
        
        
    
    def generate_edge_template(self):
        """Precompute edge index templates for all (vehicles, obstacles) combinations.

        Args:
            None.

        Returns:
            dict: Keys are (total_nodes, num_vehicles) tuples. Values are
                  [edges_vehicles (2, Ev), edges_obstacles (2, Eo)].
        """
        assert self.max_num_vehicles >= 1, \
               'Must have at least one vehicle!'
        
        assert self.max_num_obstacles >= 0, \
               'Number of obstacle should be positive integer!'
               
        edge_template = {}
        
        for num_vehicles in range(1, self.max_num_vehicles + 1):
            for num_obstacles in range(self.max_num_obstacles + 1):
                
                edges_vehicles = torch.tensor([[],[]],dtype=torch.int).to(self.device)
                edges_obstacles = torch.tensor([[],[]],dtype=torch.int).to(self.device)
                
                if num_vehicles > 1:
                    all_perm = list(permutations(range(num_vehicles), 2))
                    vehicle_1, vehicle_2 = zip(*all_perm)
                    vehicle_to_vehicle = torch.tensor([vehicle_1, vehicle_2]).to(self.device)
                    edges_vehicles = torch.cat((edges_vehicles, vehicle_to_vehicle),dim=-1)
                
                if num_obstacles > 0:
                    obstacles = torch.arange(num_vehicles, num_vehicles+num_obstacles).tile(num_vehicles).to(self.device)
                    vehicles = torch.arange(num_vehicles).repeat_interleave(num_obstacles).to(self.device)
                    obstacle_to_vehicle = torch.cat((obstacles[None,:], vehicles[None,:]),dim=0)
                    # vehicle_to_obstacle = torch.cat((vehicles[None,:], obstacles[None,:]),dim=0)
                    edges_obstacles = torch.cat((edges_obstacles, 
                                                 obstacle_to_vehicle, 
                                                 #  vehicle_to_obstacle,
                                                 ),dim=-1)
                
                edge_template[(num_vehicles+num_obstacles, num_vehicles)] = [edges_vehicles, edges_obstacles]
        
        return edge_template

    
    def get_edges(self, batches):
        """Assemble edge indices for a batch by offsetting precomputed templates.

        Args:
            batches (Tensor): (B, 2) with columns [total_nodes, num_vehicles].

        Returns:
            tuple[Tensor, Tensor]: (edges_vehicles, edges_obstacles), each (2, E).
        """
        edges_vehicles = torch.tensor([[],[]],dtype=torch.int).to(self.device)
        edges_obstacles = torch.tensor([[],[]],dtype=torch.int).to(self.device)
        
        batches_offset = torch.cumsum(batches[:,0],dim=0)[:-1]
        batches_offset = torch.cat((torch.tensor([0], device=self.device), batches_offset))
        
        for batch in torch.unique(batches, dim=0):
                
            index = torch.all(batches == batch, dim=-1)
            
            if torch.sum(index) == 0:
                continue
            
            offset = batches_offset[index]
            edges_batch_vehicles, edges_batch_obstacles = self.edge_template[tuple(batch.tolist())]
            
            edges_vehicles = torch.cat([edges_vehicles, (edges_batch_vehicles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
            edges_obstacles = torch.cat([edges_obstacles, (edges_batch_obstacles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
        
        return edges_vehicles, edges_obstacles

    def forward(self, x0, batches):
        """Forward pass: encode, apply graph convs, decode controls.

        In training mode, returns controls for a single step.
        In inference mode, auto-regressively rolls out the full trajectory.

        Args:
            x0 (Tensor): Flattened node states (total_nodes, 8).
            batches (Tensor): (B, 2) matrix [total_nodes, num_vehicles].

        Returns:
            If training: list [vehicle_controls (V, 2), static_controls (O, 2)].
            If inference: tuple (controls, statics, states, edges_vehicles, edges_obstacles).
        """
        
        marks = (x0[:,-1])
        vehicles = (marks == 0)
        obstacles = (marks != 0)
        
        edges_vehicles, edges_obstacles = self.get_edges(batches)
        
        if self.no_vehicle_edges:
            edges = edges_obstacles
        else:
            edges = torch.cat((edges_vehicles, edges_obstacles), dim=-1)
        
        if self.mode == "training":
            
            assert self.horizon == 1, \
                "In training mode of iterative GNN, the horizon need to be 1!"
            
            x = x0 
            x = self.block0(x)
            x = self.block1(x, edges)
            x = self.block2(x, edges)
            x = self.block3(x)
            
            x = x*self.bound
            
            x_vehicles = x[vehicles]
            x_obstacles = x[obstacles]
            
            controls = [x_vehicles, x_obstacles]
            
            return controls
                
        else:
                                    
            states = torch.empty((0, torch.sum(batches[:,0]), 8), device=self.device)
            states = torch.cat((states, x0[None,...]))
            controls = torch.empty((0, torch.sum(batches[:,1]), 2), device=self.device)
            statics = torch.empty((0, torch.sum(batches[:,0]-batches[:,1]), 2), device=self.device)
            
            x = x0
            
            for i in range(self.horizon):

                x = self.block0(x)                 
                x = self.block1(x, edges)
                x = self.block2(x, edges)
                x = self.block3(x)
                
                x = x*self.bound
                
                controls = torch.cat((controls, x[vehicles][None,...]))
                statics = torch.cat((statics, x[obstacles][None,...]))
                
                x = torch.empty(x0.shape, device=self.device)
                x[obstacles] = x0[obstacles]
                x[vehicles,4:] = x0[vehicles,4:]
                x[vehicles,:4] = self.vehicle_dynamic(states[-1, vehicles, :4], controls[-1])
                states = torch.cat((states, x[None,...]))
            
            controls = torch.transpose(controls, 0, 1).reshape(-1, self.horizon*2)
            statics = torch.transpose(statics, 0, 1).reshape(-1, self.horizon*2)
            states = torch.transpose(states, 0, 1)
        
            return controls, statics, states, edges_vehicles, edges_obstacles
    
    def vehicle_dynamic(self, state, control):
        """Bicycle kinematics model: update x, y, heading, velocity from control inputs.

        Args:
            state (Tensor): Current state (N, 4) = [x, y, psi, v].
            control (Tensor): Control inputs (N, 2) = [pedal, steering_angle].

        Returns:
            Tensor: Next state (N, 4) = [x, y, psi, v].
        """
        x_t = state[:,0]+state[:,3]*torch.cos(state[:,2])*self.dt
        y_t = state[:,1]+state[:,3]*torch.sin(state[:,2])*self.dt
        psi_t = state[:,2]+state[...,3]*self.dt*torch.tan(control[:,1])/2.0
        psi_t = (psi_t + pi)%(2*pi) - pi
        v_t = 0.99*state[:,3]+control[:,0]*self.dt
        
        return torch.cat((x_t[...,None], y_t[...,None], psi_t[...,None], v_t[...,None]), dim=-1)

    def forward_show_attention(self, x0, batches):
        """Forward pass returning attention logits for visualization.

        Only supports single-batch inference with UAttentionConv type.

        Args:
            x0 (Tensor): Initial states (total_nodes, 8).
            batches (Tensor): (1, 2) = [total_nodes, num_vehicles].

        Returns:
            tuple: (controls, statics, states, edges_vehicles, edges_obstacles, attention),
                   where attention is (4, V, N) containing alpha logits from 4 conv layers.
        """
        
        assert (len(batches) == 1) and self.conv_type == "UAttentionConv", \
            "Forward_show_attention only support UAttentionConv with batch_size == 1!"
        
        marks = (x0[:,-1])
        vehicles = (marks == 0)
        obstacles = (marks != 0)
        
        edges_vehicles, edges_obstacles = self.get_edges(batches)
        edges = torch.cat((edges_vehicles, edges_obstacles), dim=-1)        
         
        states = torch.empty((0, torch.sum(batches[:,0]), 8), device=self.device)
        states = torch.cat((states, x0[None,...]))
        controls = torch.empty((0, torch.sum(batches[:,1]), 2), device=self.device)
        statics = torch.empty((0, torch.sum(batches[:,0]-batches[:,1]), 2), device=self.device)
        
        x = x0
        
        x = self.block0(x)                 
        x = self.block1(x, edges)
        x = self.block2(x, edges)
        x = self.block3(x)
        
        x = x*self.bound
        
        if self.max_num_vehicles+self.max_num_obstacles > 1:
            attention = torch.zeros((4, torch.amax(batches[:,1]), torch.amax(batches[:,0])), device=self.device)
            attention[0,edges[1], edges[0]] = self.block1.conv1._alpha_logits.clone()
            attention[1,edges[1], edges[0]] = self.block1.conv2._alpha_logits.clone()
            attention[2,edges[1], edges[0]] = self.block2.conv1._alpha_logits.clone()
            attention[3,edges[1], edges[0]] = self.block2.conv2._alpha_logits.clone()
            

        controls = torch.cat((controls, x[vehicles][None,...]))
        statics = torch.cat((statics, x[obstacles][None,...]))
        
        x = torch.empty(x0.shape, device=self.device)
        x[obstacles] = x0[obstacles]
        x[vehicles,4:] = x0[vehicles,4:]
        x[vehicles,:4] = self.vehicle_dynamic(states[-1, vehicles, :4], controls[-1])
        states = torch.cat((states, x[None,...]))
        
        controls = torch.transpose(controls, 0, 1).reshape(-1, 2)
        statics = torch.transpose(statics, 0, 1).reshape(-1, 2)
        states = torch.transpose(states, 0, 1)
    
        return controls, statics, states, edges_vehicles, edges_obstacles, attention
