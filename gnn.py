import torch
from torch.nn import ReLU, Tanh, BatchNorm1d
from numpy import pi
from torch_geometric.nn import Linear, TransformerConv, EdgeConv
from itertools import permutations

from my_transformer_conv import MyTransformerConv
from graph_attention_isomorphism_net import GAINet



class ConvResidualBlock(torch.nn.Module):
    def __init__(self, io_node_num, hidden_node_num, key_query_len=512, conv_type="MyTransformerConv"):
        super().__init__()
        
        if conv_type == "MyTransformerConv":
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
        
        x = self.conv1(x0, edges)
        x = self.bn1(x)
        x = self.a1(x)
        
        x = self.conv2(x, edges)
        x = self.bn2(x)
        x = x+x0
        x = self.a2(x)
                
        return x


class LinearBlock(torch.nn.Module):
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
        
        x = self.linear(x)
        x = self.bn(x)
        
        if x0 is not None:
            x=x+x0
        
        x = self.a(x)
        
        return x


class LinearResidualBlock(torch.nn.Module):
    def __init__(self, io_node_num, hidden_node_num):
        super().__init__()
        
        self.linear1 = LinearBlock(io_node_num, hidden_node_num)
        self.linear2 = LinearBlock(hidden_node_num, io_node_num)
    
        
    def forward(self, x0):
        
        x = self.linear1(x0)
        x = self.linear2(x,x0)
        
        return x
       

class IterativeGNNModel(torch.nn.Module):
    def __init__(self, horizon, max_num_vehicles, max_num_obstacles, use_relative_frame = False, device = 'cpu',
                 mode = "inference", conv_type="MyTransformerConv"):
        super().__init__()
        self.device = device
        self.horizon = horizon
        self.dt = 0.2
        self.input_length = 5 if use_relative_frame else 8
        self.output_length = 2
        self.bound = torch.tensor([1, 0.8]).to(self.device)
        self.use_relative_frame = use_relative_frame
        self.mode = mode
        self.conv_type = conv_type
            
        self.max_num_vehicles = max_num_vehicles
        self.max_num_obstacles = max_num_obstacles
        
        if use_relative_frame:
            assert self.max_num_vehicles == 1, \
                   "relative frame can only be used in one vehicle case!"
        
        self.edge_template = self.generate_edge_template()
        
        self.block0 = LinearBlock(self.input_length,80)
        self.block1 = ConvResidualBlock(80,160,conv_type=conv_type)
        self.block2 = ConvResidualBlock(80,160,conv_type=conv_type)
        self.block3 = LinearBlock(80, self.output_length, activation="tanh")  
    
    def generate_edge_template(self):
        
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
        
        marks = (x0[:,-1])
        vehicles = (marks == 0)
        obstacles = (marks != 0)
        
        edges_vehicles, edges_obstacles = self.get_edges(batches)
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
                
                if self.use_relative_frame:
                    pass

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
        x_t = state[:,0]+state[:,3]*torch.cos(state[:,2])*self.dt
        y_t = state[:,1]+state[:,3]*torch.sin(state[:,2])*self.dt
        psi_t = state[:,2]+state[...,3]*self.dt*torch.tan(control[:,1])/2.0
        psi_t = (psi_t + pi)%(2*pi) - pi
        v_t = 0.99*state[:,3]+control[:,0]*self.dt
        
        return torch.cat((x_t[...,None], y_t[...,None], psi_t[...,None], v_t[...,None]), dim=-1)

    def forward_show_attention(self, x0, batches):
        
        assert (len(batches) == 1) and self.conv_type == "MyTransformerConv", \
            "Forward_show_attention only support MyTransformerConv with batch_size == 1!"
        
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