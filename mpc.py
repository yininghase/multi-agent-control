import numpy as np
from numpy import pi
from itertools import permutations
from data_process import get_angle_diff

class ModelPredictiveControl:
    """MPC controller: wraps cost function, dynamics model, and obstacle/collision penalties for trajectory optimization.

    The cost function includes position error, angle error, obstacle avoidance,
    control smoothness, travel distance, and vehicle collision penalties.

    Args:
        simulation_options (dict): Configuration dictionary containing horizon,
            start, target, obstacles, cost weights, and radii.
    """
    def __init__(self, simulation_options):
        """Initialize the Model Predictive Controller.

        Args:
            simulation_options (dict): Configuration dictionary containing horizon, start, target,
                                       obstacles, control_init, cost weights, and geometry parameters.

        Returns:
            None.
        """
        self.horizon = simulation_options["horizon"]
        self.dt = 0.2
        self.start = simulation_options["start"]
        self.target = simulation_options["target"]
        self.obstacles = simulation_options["obstacles"] 
        self.control_init = simulation_options["control init"] 
        self.dis_cost = simulation_options["distance cost"] 
        self.ang_cost = simulation_options["angle cost"] 
        self.col_cost = simulation_options["collision cost"] 
        self.col_radius = simulation_options["collision radius"] 
        self.obs_cost = simulation_options["obstacle cost"] 
        self.obs_radius = simulation_options["obstacle radius"]
        
        self.smooth_cost = simulation_options["smoothness cost"]
        self.travel_dist_cost = simulation_options["travel dist cost"]

        
        assert len(self.start) == len(self.target)
        self.num_vehicle = len(self.start)
        self.num_obstacle = len(self.obstacles)
        
        self.vehicle_pair = np.array(list(permutations(range(self.num_vehicle), 2)))
        # print()

    def plant_model(self, prev_state, dt, control):
        """Bicycle kinematics model: compute next state from previous state and control inputs.

        Args:
            prev_state (ndarray): Current state (N, 4) = [x, y, psi, v].
            dt (float): Time step.
            control (ndarray): Control inputs (N, 2) = [pedal, steering_angle].

        Returns:
            ndarray: Next state (N, 4) = [x, y, psi, v].
        """
        
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        
        pedal = control[:,0]
        steering = control[:,1]

        # Vehicle Dynamic Equation
        x_t = x_t+v_t*np.cos(psi_t)*dt
        y_t = y_t+v_t*np.sin(psi_t)*dt
        psi_t = psi_t+v_t*dt*np.tan(steering)/2.0
        psi_t = (psi_t+pi)%(2*pi)-pi
        v_t = 0.99*v_t+pedal*dt
        
        next_state = np.concatenate((x_t[...,None], 
                                     y_t[...,None], 
                                     psi_t[...,None], 
                                     v_t[...,None]), 
                                    axis=-1)

        return next_state

    def cost_function(self, u, *args):
        """Compute total cost: position error + angle error + obstacle penalty + smoothness + travel distance + collision.

        Args:
            u (ndarray): Flattened control sequence (horizon * num_vehicle * 2,).
            *args (tuple): Additional arguments: (initial_state, reference_targets).

        Returns:
            float: Total scalar cost.
        """
        u = u.reshape(self.horizon, self.num_vehicle, 2)
        
        state = np.array(args[0])
        ref = np.array(args[1])
        
        state_history =np.array([state])
        
        for i in range(self.horizon):
            control = u[i]
            state = self.plant_model(state, self.dt, control)
            state_history= np.concatenate((state_history, state[None,...]))

        # target cost
        pos_diff = np.linalg.norm(ref[:,:2]-state_history[:,:,:2], axis=-1, ord=2)        
        cost = np.sum(pos_diff[1:])*self.dis_cost 
        
        angle_diff = get_angle_diff(state_history[1:,:,2], ref[:,2]) 
        cost += np.sum(angle_diff)*self.ang_cost
        
        # obstale cost
        if self.obs_cost > 0 and self.num_obstacle > 0:
            dist = np.linalg.norm(self.obstacles[:,:2]-state_history[1:,:,None,:2], axis=-1, ord=2)-self.obstacles[:,2]
            dist = np.clip(dist, a_min=0, a_max=None) + 1e-8
            cost += np.sum((1/dist - 1/self.obs_radius) * (dist < self.obs_radius)) * self.obs_cost
        
        # smoothness cost (steering angle and pedal input)
        if self.smooth_cost > 0:
            cost += np.sum(np.diff(u, axis=0)) * self.smooth_cost
        
        # travel distance cost
        if self.travel_dist_cost > 0:
            cost += np.sum(np.linalg.norm(np.diff(state_history[...:2], axis=0), ord=2, axis=-1)) * self.travel_dist_cost
        
        # collision cost
        if self.col_cost > 0 and self.num_vehicle > 1:
            dist = np.linalg.norm(state_history[1:, self.vehicle_pair[:,1], :2] - state_history[1:, self.vehicle_pair[:,0], :2], ord=2, axis=-1)            
            cost += np.sum((1/dist-1/self.col_radius) * (dist < self.col_radius))*self.col_cost            
            
        return cost
    
