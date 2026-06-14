import os
import torch

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

from calculate_metrics import (check_collision_rectangular_circle, 
                               check_collision_rectangular_rectangular)


class Visualize_Trajectory:
    """Visualizes vehicle trajectories as an animation, including predicted paths and collision markers."""
    def __init__(self, simulation_options, show_attention=False):
        """Initialize visualization with simulation options.

        Args:
            simulation_options (dict): Configuration with plot, vehicle, and obstacle parameters.
            show_attention (bool, optional): Enable attention subplot. Defaults to False.

        Returns:
            None.
        """
        self.simulation_options = simulation_options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5),
                     ]
        self.show_attention = show_attention

    
    def base_plot(self, is_trajectory):
        """Set up the base matplotlib figure with vehicles, targets, and obstacles.

        Args:
            is_trajectory (bool): If True, only plot (no legend); if False, animation mode with legend.

        Returns:
            None.
        """
        
        if self.show_attention:
            self.fig = plt.figure(figsize=(2*self.simulation_options["figure size"], 
                                        self.simulation_options["figure size"]))
            self.ax = self.fig.add_subplot(1,2,1)
            self.ax_ = self.fig.add_subplot(1,2,2)
            
        else:
            self.fig = plt.figure(figsize=(self.simulation_options["figure size"], 
                                        self.simulation_options["figure size"]))
            self.ax = self.fig.add_subplot()
            
        self.ax.set_xlim(-self.simulation_options["figure limit"], 
                 self.simulation_options["figure limit"])
        self.ax.set_ylim([-self.simulation_options["figure limit"], 
                          self.simulation_options["figure limit"]])
        self.ax.set_xticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))
        
        self.ax.set_yticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))

        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.patch_target = []
        self.patch_target_arrow = []
        self.predicts_opt = []
        self.predicts_model = []

        patch_obs = []
        
        # start: [num_vehicle, [x, y, psi]]
        # target: [num_vehicle, [x, y, psi]]
        start = self.simulation_options["start"]
        target = self.simulation_options["target"]
        
        start_new = self.car_patch_pos(self.simulation_options["start"])
        target_new = self.car_patch_pos(self.simulation_options["target"])

        for i in range(self.simulation_options["num of vehicles"]):
            
            patch_car = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i])
            patch_car.set_xy(start_new[i,:2])
            patch_car.angle = np.rad2deg(start_new[i,2])-90
            self.patch_vehicles.append(patch_car)
            
            patch_car_arrow = mpatches.FancyArrow(start[i,0]-0.9*np.cos(start[i,2]), 
                                                  start[i,1]-0.9*np.sin(start[i,2]), 
                                                  1.5*np.cos(start[i,2]), 
                                                  1.5*np.sin(start[i,2]), 
                                                  width=0.1, color='w')
            self.patch_vehicles_arrow.append(patch_car_arrow)

            patch_goal = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            
            patch_goal.set_xy(target_new[i,:2])
            patch_goal.angle = np.rad2deg(target_new[i,2])-90
            self.patch_target.append(patch_goal)
            
            patch_goal_arrow = mpatches.FancyArrow(target[i,0]-0.9*np.cos(target[i,2]), 
                                                   target[i,1]-0.9*np.sin(target[i,2]), 
                                                   1.5*np.cos(target[i,2]), 
                                                   1.5*np.sin(target[i,2]), 
                                                   width=0.1, 
                                                   color=self.cmap[i])
            self.patch_target_arrow.append(patch_goal_arrow)

            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)
            self.ax.add_patch(patch_car_arrow)
            self.ax.add_patch(patch_goal_arrow)

            self.frame = plt.text(12, 12, "", fontsize=15)

            if self.simulation_options["show optimization"]:
                if is_trajectory:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1)
                elif i == 0:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="Optimization")
                else:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="_Optimization")
                self.predicts_opt.append(predict_opt)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"vehicle {i+1}")
        
        for i, obs in enumerate(self.simulation_options["obstacles"]):
            patch_obs.append(mpatches.Circle(obs[:2], obs[2], color=self.cmap[i], fill=True))
            self.ax.add_patch(patch_obs[-1])
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"obstacle {i+1}")
        
        if not is_trajectory:   
            self.ax.legend(loc='upper left', fontsize=12)
    
    
    def create_video(self, data, predict_opt, predict_model, attention=None):
        """Create animation from simulation data showing vehicle movement and predictions.

        Args:
            data (np.ndarray): Simulation state history (T, num_vehicles, 4).
            predict_opt (np.ndarray): MPC predictions (T, horizon+1, num_vehicles, 4).
            predict_model (np.ndarray): Model predictions (T, horizon+1, num_vehicles, 4).
            attention (np.ndarray, optional): Attention weights for display. Defaults to None.

        Returns:
            None.
        """
        self.base_plot(is_trajectory=False)
        self.data = data
        if self.simulation_options["is model"]:
            self.predict_model = predict_model
        if self.simulation_options["show optimization"]:
            self.predict_opt = predict_opt 
        self.attention = attention
            
        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(data)-1), interval=100, repeat=True, blit=False)
        
        if self.simulation_options["save plot"]:
            if not os.path.exists(self.simulation_options["plot folder"]):
                os.makedirs(self.simulation_options["plot folder"])
                
            car_animation.save(os.path.join(self.simulation_options["plot folder"], 
                                            self.simulation_options["name"] + ".gif"))
            
        if self.simulation_options["show plot"]:
            plt.show()

    def update_plot(self, num):
        """Update function for animation: redraw vehicles and predicted trajectories at each frame.

        Args:
            num (int): Frame index.

        Returns:
            None.
        """
        
        data = self.data[num,...]               
        
        for i in range(self.simulation_options["num of vehicles"]):
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))
            
            
            if self.simulation_options["show optimization"]:
                self.predicts_opt[i].set_data(self.predict_opt[num, :, i, 0], self.predict_opt[num, :, i, 1])
                
        
        if self.show_attention and self.attention is not None:
            self.ax_.imshow(self.attention[num], vmin=-2.5, vmax=2.5, cmap="gray")
            self.ax_.set_xticks(ticks=[i for i in range(self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])] + \
                                         [f"obstacle {i+1}" for i in range(self.simulation_options["num of obstacles"])])
            self.ax_.set_yticks(ticks=[i for i in range(self.simulation_options["num of vehicles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])])
            
    def plot_trajectory(self, points):
        """Plot full trajectory as a colored line with collision markers.

        Args:
            points (np.ndarray): Trajectory points (T, num_vehicles, 4).

        Returns:
            None.
        """
        self.base_plot(is_trajectory=True)
        max_time = points.shape[0]
        
        for i in range(self.simulation_options["num of vehicles"]):
            veh_points = points[:, i, :2][:,None,:]
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            norm = plt.Normalize(0, max_time)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(range(points.shape[0]))
            lc.set_linewidth(2)
            line = self.ax.add_collection(lc)
        
        collision = np.zeros((points.shape[0], points.shape[1]), dtype=bool)
        
        for i in range(self.simulation_options["num of vehicles"]-1):
            for j in range(i+1, self.simulation_options["num of vehicles"]):
                collisions_ij = check_collision_rectangular_rectangular(torch.from_numpy(points[:,i,:]).type(torch.float32), 
                                                                        torch.from_numpy(points[:,j,:]).type(torch.float32), 
                                                                        vehicle_size=self.simulation_options["car size"]).numpy()
                collision[collisions_ij,i]=True
                collision[collisions_ij,j]=True
        
        for i in range(self.simulation_options["num of vehicles"]):
            for j in range(self.simulation_options["num of obstacles"]):
                obstacle_j = self.simulation_options["obstacles"][j]
                obstacle_j = np.concatenate((np.zeros(4),obstacle_j,np.ones(1)))[None,:]
                
                collisions_ij = check_collision_rectangular_circle(torch.from_numpy(points[:,i,:]).type(torch.float32), 
                                                                   torch.from_numpy(obstacle_j).type(torch.float32), 
                                                                   vehicle_size=self.simulation_options["car size"]).numpy()
                
                collision[collisions_ij,i]=True
               
        cbar = self.fig.colorbar(line, ax=self.ax)
        cbar.ax.set_ylabel("Timestep", fontsize=15)
        
        if np.sum(collision)>0:
            self.ax.scatter(points[collision][:,0], points[collision][:,1], s=15, c="r", marker="o")
        
        if self.simulation_options["save plot"]:
            if not os.path.exists(self.simulation_options["plot folder"]):
                os.makedirs(self.simulation_options["plot folder"])
            plt.savefig(os.path.join(self.simulation_options["plot folder"], 
                                     self.simulation_options["name"]+".png"), bbox_inches='tight')
        
        if self.simulation_options["show plot"]:
            plt.show()

    
    def car_patch_pos(self, posture):
        """Convert vehicle posture to bottom-left corner position for matplotlib Rectangle patch.

        Args:
            posture (np.ndarray): Vehicle state(s) (..., 3) = [x, y, heading].

        Returns:
            np.ndarray: Transformed posture (bottom-left x, bottom-left y, heading).
        """
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.simulation_options["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.simulation_options["car size"][1]/2)
        
        return posture_new
    
    def calculate_cost(self, coordinates, targets):
        """Compute heatmap cost combining distance to goal and obstacle proximity.

        Args:
            coordinates (np.ndarray): Grid coordinates (resolution, resolution, 2).
            targets (np.ndarray): Target positions (num_vehicles, 2).

        Returns:
            np.ndarray: Cost map of shape (resolution, resolution).
        """
        
        dist_cost = self.simulation_options["distance_cost"]
        obst_cost = self.simulation_options["obstacle_cost"]
        obs_radius = self.simulation_options["obstacle_radius"]
        obstacles = self.simulation_options["obstacles"]
        num_obstacles = self.simulation_options["num of obstacles"]
        
        loss = np.linalg.norm(coordinates - targets[None, None, :2], ord=2, axis=-1)*dist_cost
        
        if  obst_cost > 0 and num_obstacles > 0:
            dist = np.linalg.norm(coordinates[:,:,None,:]-obstacles[None,None,:,:2], ord=2, axis=-1)-obstacles[None,None,:,2]-obs_radius
            dist = (np.clip(-dist, a_min=0, a_max=None))**2
            loss += np.sum(dist, axis=-1) * obst_cost
        
        return loss


class Visualize_Attention:
    """Interactive visualization tool for inspecting attention weights of the U-Attention GNN model."""
    def __init__(self, simulation_options, model, device):
        """Initialize attention visualization with model and device.

        Args:
            simulation_options (dict): Configuration with plot, vehicle, and obstacle parameters.
            model (IterativeGNNModel): Trained GNN model.
            device (str): Device for model inference ('cpu' or 'cuda').

        Returns:
            None.
        """
        self.simulation_options = simulation_options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5),
                     ]

        self.model = model
        self.device = device
    
    def base_plot(self, move_mode=False):
        """Set up the base figure with vehicle/obstacle patches, optional sliders for velocity, and mouse interaction.

        Args:
            move_mode (bool, optional): If True, add velocity sliders and mouse interaction. Defaults to False.

        Returns:
            None.
        """
        
        self.fig = plt.figure(figsize=(2*self.simulation_options["figure size"], 
                                    self.simulation_options["figure size"]))
        self.ax = self.fig.add_subplot(1,2,1)
        self.ax_ = self.fig.add_subplot(1,2,2)
            
        self.ax.set_xlim(-self.simulation_options["figure limit"], 
                 self.simulation_options["figure limit"])
        self.ax.set_ylim([-self.simulation_options["figure limit"], 
                          self.simulation_options["figure limit"]])
        self.ax.set_xticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))
        
        self.ax.set_yticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))

        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.patch_target = []
        self.patch_target_arrow = []
        self.predicts_opt = []
        self.predicts_model = []

        self.patch_obs = []
        
        # start: [num_vehicle, [x, y, psi]]
        # target: [num_vehicle, [x, y, psi]]
        start = self.simulation_options["start"]
        target = self.simulation_options["target"]
        
        start_new = self.car_patch_pos(self.simulation_options["start"])
        target_new = self.car_patch_pos(self.simulation_options["target"])

        for i in range(self.simulation_options["num of vehicles"]):
            
            patch_car = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i])
            patch_car.set_xy(start_new[i,:2])
            patch_car.angle = np.rad2deg(start_new[i,2])-90
            self.patch_vehicles.append(patch_car)
            
            patch_car_arrow = mpatches.FancyArrow(start[i,0]-0.9*np.cos(start[i,2]), 
                                                  start[i,1]-0.9*np.sin(start[i,2]), 
                                                  1.5*np.cos(start[i,2]), 
                                                  1.5*np.sin(start[i,2]), 
                                                  width=0.1, color='w')
            self.patch_vehicles_arrow.append(patch_car_arrow)

            patch_goal = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            
            patch_goal.set_xy(target_new[i,:2])
            patch_goal.angle = np.rad2deg(target_new[i,2])-90
            self.patch_target.append(patch_goal)
            
            patch_goal_arrow = mpatches.FancyArrow(target[i,0]-0.9*np.cos(target[i,2]), 
                                                   target[i,1]-0.9*np.sin(target[i,2]), 
                                                   1.5*np.cos(target[i,2]), 
                                                   1.5*np.sin(target[i,2]), 
                                                   width=0.1, 
                                                   color=self.cmap[i])
            self.patch_target_arrow.append(patch_goal_arrow)

            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)
            self.ax.add_patch(patch_car_arrow)
            self.ax.add_patch(patch_goal_arrow)

            self.frame = plt.text(12, 12, "", fontsize=15)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"vehicle {i+1}")
        
        for i, obs in enumerate(self.simulation_options["obstacles"]):
            self.patch_obs.append(mpatches.Circle(obs[:2], obs[2], color=self.cmap[i], fill=True))
            self.ax.add_patch(self.patch_obs[-1])
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"obstacle {i+1}")
        
          
        self.ax.legend(loc='upper left', fontsize=12)

        if move_mode:
            
            class Update:
                def __init__(self, index, root):
                    """Initialize the slider update handler.

                    Args:
                        index (int): Vehicle index this slider controls.
                        root (Visualize_Attention): Parent visualization instance.

                    Returns:
                        None.
                    """
                    self.index = index
                    self.root = root
                    self.simulation_options = self.root.simulation_options
                
                def update(self, val):
                    """Callback for slider value change: recompute attention.

                    Args:
                        val (float): New velocity value from slider.

                    Returns:
                        None.
                    """
                    self.simulation_options["start"][self.index, 3] = val
                    vehicle_input = np.concatenate((self.simulation_options["start"], self.simulation_options["target"], 
                                            np.zeros((self.simulation_options["num of vehicles"], 1))), axis=1)
                    obstacle_input = np.concatenate((self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 2)), 
                                                    self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 1)),
                                                    self.simulation_options["obstacles"][:,2:3], ), axis=1)
                    model_input = np.concatenate((vehicle_input, obstacle_input), axis=0)
                    batches = np.array([[self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"], 
                                        self.simulation_options["num of vehicles"]]])
                    
                    model_input = torch.from_numpy(model_input).type(torch.float32).to(self.root.device)
                    batches = torch.from_numpy(batches).type(torch.long).to(self.root.device)
                    
                    _, _, _, _, _, attention = self.root.model.forward_show_attention(model_input, batches)
                    attention = np.mean(attention.clone().detach().cpu().numpy(), axis=0)
                    
                    self.root.show_attention(attention)
                    self.root.fig.canvas.draw()
                    
            
            self.velocity_slider = []
            self.update = []
            
            tmp_var = self.simulation_options["num of vehicles"]+2
            
            self.fig.subplots_adjust(bottom=0.03*tmp_var)
            
            for i in range(self.simulation_options["num of vehicles"]):
                ax_vel = self.fig.add_axes([0.1, 0.03*(tmp_var-i-2), 0.8, 0.01])
                self.velocity_slider.append(Slider(
                    ax=ax_vel,
                    label = f"velocity vehicle {i+1}",
                    valmin=-2,
                    valmax=3,
                    valinit=self.simulation_options["start"][i, 3],
                    color=self.cmap[i]
                ))       
                                
                self.update.append(Update(i, self))
                self.velocity_slider[i].on_changed(self.update[i].update)
            
            
            vehicle_input = np.concatenate((self.simulation_options["start"], self.simulation_options["target"], 
                                            np.zeros((self.simulation_options["num of vehicles"], 1))), axis=1)
            obstacle_input = np.concatenate((self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 2)), 
                                             self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 1)),
                                             self.simulation_options["obstacles"][:,2:3], ), axis=1)
            model_input = np.concatenate((vehicle_input, obstacle_input), axis=0)
            batches = np.array([[self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"], 
                                 self.simulation_options["num of vehicles"]]])
            
            model_input = torch.from_numpy(model_input).type(torch.float32).to(self.device)
            batches = torch.from_numpy(batches).type(torch.long).to(self.device)
                
            _, _, _, _, _, attention = self.model.forward_show_attention(model_input, batches)
            attention = np.mean(attention.clone().detach().cpu().numpy(), axis=0)
            
            self.show_attention(attention)
            self.fig.canvas.draw()
            
            self.move_object = None
            self.move_object_type = None
            self.move_index = None
            self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            
            plt.show()


    def show_attention(self, attention):       
        """Display attention weight matrix as a grayscale image with labeled axes.

        Args:
            attention (np.ndarray): Attention weight matrix (num_vehicles, total_nodes).

        Returns:
            None.
        """
        
        self.ax_.imshow(attention, vmin=-2.5, vmax=2.5, cmap="gray")
        self.ax_.set_xticks(ticks=[i for i in range(self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"])],
                            labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])] + \
                                        [f"obstacle {i+1}" for i in range(self.simulation_options["num of obstacles"])])
        self.ax_.set_yticks(ticks=[i for i in range(self.simulation_options["num of vehicles"])],
                            labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])])
        
        for i in range(self.simulation_options["num of vehicles"]):
            self.ax_.get_xticklabels()[i].set_color(self.cmap[i])
            self.ax_.get_yticklabels()[i].set_color(self.cmap[i])
        
        for i in range(self.simulation_options["num of obstacles"]):
            self.ax_.get_xticklabels()[i+self.simulation_options["num of vehicles"]].set_color(self.cmap[i])        
    

    def car_patch_pos(self, posture):
        """Convert vehicle posture to bottom-left corner position for matplotlib Rectangle patch.

        Args:
            posture (np.ndarray): Vehicle state(s) (..., 3) = [x, y, heading].

        Returns:
            np.ndarray: Transformed posture (bottom-left x, bottom-left y, heading).
        """
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.simulation_options["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.simulation_options["car size"][1]/2)
        
        return posture_new
    
    
    def on_press(self, event):
        """Handle mouse press: identify clicked object (vehicle, target, obstacle).

        Args:
            event (MouseEvent): Matplotlib mouse press event.

        Returns:
            None.
        """
        
        for i in range(self.simulation_options["num of vehicles"]):
            move_vehicle = self.patch_vehicles[i]
            move_arrow = self.patch_vehicles_arrow[i]
        
            if move_vehicle.contains(event)[0]:
                self.move_object = [move_vehicle, move_arrow]
                self.move_object_type = "vehicle"
                self.move_index = i
                self.fig.canvas.draw()
                return
        
        for i in range(self.simulation_options["num of vehicles"]):
            move_vehicle = self.patch_target[i]
            move_arrow = self.patch_target_arrow[i]
        
            if move_vehicle.contains(event)[0]:
                self.move_object = [move_vehicle, move_arrow]
                self.move_object_type = "target"
                self.move_index = i
                self.fig.canvas.draw()
                return
            
        for i in range(self.simulation_options["num of obstacles"]):
            move_obstacle = self.patch_obs[i]
        
            if move_obstacle.contains(event)[0]:
                self.move_object = [move_obstacle]
                self.move_object_type = "obstacle"
                self.move_index = i
                self.fig.canvas.draw()
                return
            

    def on_motion(self, event):
        """Handle mouse drag: update position of vehicle, target, or obstacle.

        Args:
            event (MouseEvent): Matplotlib mouse motion event.

        Returns:
            None.
        """
        if self.move_object_type == "vehicle":
            vehicle = self.move_object[0]
            arrow = self.move_object[1]
            posture = np.array([event.xdata, event.ydata, 
                                np.deg2rad(vehicle.angle+90)])
            posture_ = self.car_patch_pos(posture)
            
            vehicle.set_xy(posture_[:2])            
            arrow.set_data(x = posture[0]-0.9*np.cos(posture[2]),
                           y = posture[1]-0.9*np.sin(posture[2]),
                           dx = 1.5*np.cos(posture[2]),
                           dy = 1.5*np.sin(posture[2]))
            
            i = self.move_index
            self.simulation_options["start"][i,:2] = posture[:2] 
            
            self.fig.canvas.draw()
            
        elif self.move_object_type == "target":
            vehicle = self.move_object[0]
            arrow = self.move_object[1]
            posture = np.array([event.xdata, event.ydata, 
                                np.deg2rad(vehicle.angle+90)])
            posture_ = self.car_patch_pos(posture)
            
            vehicle.set_xy(posture_[:2])            
            arrow.set_data(x = posture[0]-0.9*np.cos(posture[2]),
                           y = posture[1]-0.9*np.sin(posture[2]),
                           dx = 1.5*np.cos(posture[2]),
                           dy = 1.5*np.sin(posture[2]))
            
            i = self.move_index
            self.simulation_options["target"][i,:2] = posture[:2] 
            
            self.fig.canvas.draw()
            
        elif self.move_object_type == "obstacle":
            obstacle = self.move_object[0]
            obstacle.set_center((event.xdata, event.ydata))
            i = self.move_index
            self.simulation_options["obstacles"][i,:2] = np.array([event.xdata, event.ydata])
            
            self.fig.canvas.draw()


    def on_scroll(self, event):
        """Handle scroll: rotate vehicle/target heading or resize obstacle radius.

        Args:
            event (MouseEvent): Matplotlib scroll event.

        Returns:
            None.
        """
        if self.move_object_type == "vehicle":
            vehicle = self.move_object[0]
            arrow = self.move_object[1]
            i = self.move_index
            
            posture = self.simulation_options["start"][i, :3].copy()
            
            if event.button == 'up':
                angle = posture[2] + np.pi/18   
            elif event.button == 'down':
                angle = posture[2] - np.pi/18  
                
            posture[2] = (angle+np.pi)%(2*np.pi)- np.pi
            posture_ = self.car_patch_pos(posture)
            
            vehicle.set_xy(posture_[:2])
            vehicle.angle = np.rad2deg(posture[2])-90      
            arrow.set_data(x = posture[0]-0.9*np.cos(posture[2]),
                           y = posture[1]-0.9*np.sin(posture[2]),
                           dx = 1.5*np.cos(posture[2]),
                           dy = 1.5*np.sin(posture[2]))
            
            
            self.simulation_options["start"][i,2] = posture[2] 
            
            self.fig.canvas.draw()
            
        elif self.move_object_type == "target":
            vehicle = self.move_object[0]
            arrow = self.move_object[1]
            i = self.move_index
            
            posture = self.simulation_options["target"][i, :3].copy()
            
            if event.button == 'up':
                angle = posture[2] + np.pi/18   
            elif event.button == 'down':
                angle = posture[2] - np.pi/18  
                
            posture[2] = (angle+np.pi)%(2*np.pi)- np.pi
            posture_ = self.car_patch_pos(posture)
            
            vehicle.set_xy(posture_[:2])
            vehicle.angle = np.rad2deg(posture[2])-90             
            arrow.set_data(x = posture[0]-0.9*np.cos(posture[2]),
                           y = posture[1]-0.9*np.sin(posture[2]),
                           dx = 1.5*np.cos(posture[2]),
                           dy = 1.5*np.sin(posture[2]))
            
            
            self.simulation_options["target"][i,2] = posture[2] 
            
            self.fig.canvas.draw()
            
        elif self.move_object_type == "obstacle":
            obstacle = self.move_object[0]
            if event.button == 'up':
                radius = obstacle.radius + 0.1   
            elif event.button == 'down':
                radius = obstacle.radius - 0.1
            radius = np.clip(float(radius), a_min=1, a_max=4)  
            
            obstacle.set_radius(radius) 
            
            i = self.move_index
            self.simulation_options["obstacles"][i,2] = radius 
            
            self.fig.canvas.draw()


    def on_release(self, event):
        """Handle mouse release: update attention visualization after drag.

        Args:
            event (MouseEvent): Matplotlib mouse release event.

        Returns:
            None.
        """
        if self.move_object_type is not None:
            
            self.move_object = None
            self.move_object_type = None
            self.move_index = None
            
            vehicle_input = np.concatenate((self.simulation_options["start"], self.simulation_options["target"], 
                                            np.zeros((self.simulation_options["num of vehicles"], 1))), axis=1)
            obstacle_input = np.concatenate((self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 2)), 
                                             self.simulation_options["obstacles"][:,:2], np.zeros((self.simulation_options["num of obstacles"], 1)),
                                             self.simulation_options["obstacles"][:,2:3], ), axis=1)
            model_input = np.concatenate((vehicle_input, obstacle_input), axis=0)
            batches = np.array([[self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"], 
                                 self.simulation_options["num of vehicles"]]])
            
            model_input = torch.from_numpy(model_input).type(torch.float32).to(self.device)
            batches = torch.from_numpy(batches).type(torch.long).to(self.device)
            
            _, _, _, _, _, attention = self.model.forward_show_attention(model_input, batches)
            attention = np.mean(attention.clone().detach().cpu().numpy(), axis=0)
            
            self.show_attention(attention)
            self.fig.canvas.draw()
