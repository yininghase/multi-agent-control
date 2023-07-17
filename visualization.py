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
    def __init__(self, simulation_options, show_attention=False):
        
        self.simulation_options = simulation_options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5),
                     ]
        self.show_attention = show_attention

    # start: [num_vehicle, [x, y, psi]]
    # target: [num_vehicle, [x, y, psi]]
    
    def base_plot(self, is_trajectory):
        
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
        
        start = self.simulation_options["start"]
        target = self.simulation_options["target"]
        
        start_new = self.car_patch_pos(self.simulation_options["start"])
        target_new = self.car_patch_pos(self.simulation_options["target"])

        for i in range(self.simulation_options["num of vehicles"]):
            # cars
            
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

            # trajectories
            if self.simulation_options["show optimization"]:
                if is_trajectory:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1)
                elif i == 0:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="Optimization")
                else:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="_Optimization")
                self.predicts_opt.append(predict_opt)
            
            # if self.simulation_options["is model"]:
            #     if is_trajectory:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1)
            #     elif i == 0:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="Model Prediction")
            #     else:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="_Model Prediction")
            #     self.predicts_model.append(predict_model)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"vehicle {i+1}")
        
        for i, obs in enumerate(self.simulation_options["obstacles"]):
            patch_obs.append(mpatches.Circle(obs[:2], obs[2], color=self.cmap[i], fill=True))
            self.ax.add_patch(patch_obs[-1])
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"obstacle {i+1}")
        
        if not is_trajectory:   
            self.ax.legend(loc='upper left', fontsize=12)
    
    
    def create_video(self, data, predict_opt, predict_model, attention=None):
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
        
        data = self.data[num,...]               
        
        # self.frame.set_text("Frame: " + str(num))
        for i in range(self.simulation_options["num of vehicles"]):
            # vehicle
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))
            
            
            if self.simulation_options["show optimization"]:
                self.predicts_opt[i].set_data(self.predict_opt[num, :, i, 0], self.predict_opt[num, :, i, 1])
                
                
            # if self.simulation_options["is model"]:
            #     self.predicts_model[i].set_data(self.predict_model[num, :, i, 0], self.predict_model[num, :, i, 1])
        
            # self.velocities[i].set_text("Velocity: " + str(round(self.data[num, i, 3], 2)))
        
        
        if self.show_attention and self.attention is not None:
            self.ax_.imshow(self.attention[num], vmin=-2.5, vmax=2.5, cmap="gray")
            self.ax_.set_xticks(ticks=[i for i in range(self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])] + \
                                         [f"obstacle {i+1}" for i in range(self.simulation_options["num of obstacles"])])
            self.ax_.set_yticks(ticks=[i for i in range(self.simulation_options["num of vehicles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])])
            
    def plot_trajectory(self, points):
        self.base_plot(is_trajectory=True)
        max_time = points.shape[0]
        
        for i in range(self.simulation_options["num of vehicles"]):
            veh_points = points[:, i, :2][:,None,:]
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            norm = plt.Normalize(0, max_time)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            # lc = LineCollection(segments, colors=self.cmap[i])
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
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.simulation_options["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.simulation_options["car size"][1]/2)
        
        return posture_new
    
    """
    Creates a heatmap
    state_data: dims (simulation_length, num_vehicles, 4)
    """
    def calculate_cost(self, coordinates, targets):
        
        dist_cost = self.simulation_options["distance_cost"]
        obst_cost = self.simulation_options["obstacle_cost"]
        obs_radius = self.simulation_options["obstacle_radius"]
        obstacles = self.simulation_options["obstacles"]
        num_obstacles = self.simulation_options["num of obstacles"]
        
        loss = np.linalg.norm(coordinates - targets[None, None, :2], ord=2, axis=-1)*dist_cost
        
        if  obst_cost > 0 and num_obstacles > 0:
            
            # dist = torch.norm(preds[:,1:,:,:2]-obs[:,None,None,:2], dim=-1, p=2)-obs[:,None,None,2]
            # dist1 = torch.clip(dist, min=0, max=None) + 1e-8
            # loss += torch.sum((1/dist1 - 1/self.obs_radius) * (dist1 < self.obs_radius), dim=(-1,-2)) * self.obs_cost
            # # dist2 = (torch.clip(-dist, min=0, max=None) + 100)**4 - 1e8
            # dist2 = torch.exp(torch.clip(-dist, min=0, max=None) + 10) - np.exp(10)
            # loss += torch.sum(dist2, dim=(-1,-2)) * self.obs_cost
            dist = np.linalg.norm(coordinates[:,:,None,:]-obstacles[None,None,:,:2], ord=2, axis=-1)-obstacles[None,None,:,2]-obs_radius
            dist = (np.clip(-dist, a_min=0, a_max=None))**2
            loss += np.sum(dist, axis=-1) * obst_cost
        
        return loss

class Visualize_Attention:
    def __init__(self, simulation_options, model, device):
        
        self.simulation_options = simulation_options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5),
                     ]

        self.model = model
        self.device = device
    # start: [num_vehicle, [x, y, psi]]
    # target: [num_vehicle, [x, y, psi]]
    
    def base_plot(self, move_mode=False):
        
        
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
        
        start = self.simulation_options["start"]
        target = self.simulation_options["target"]
        
        start_new = self.car_patch_pos(self.simulation_options["start"])
        target_new = self.car_patch_pos(self.simulation_options["target"])

        for i in range(self.simulation_options["num of vehicles"]):
            # cars
            
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
                    self.index = index
                    self.root = root
                    self.simulation_options = self.root.simulation_options
                
                def update(self, val):
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
                    # orientation="vertical"
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
            
            # self.base_plot()
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

    # The function to be called anytime a slider's value changes
   
    

    def show_attention(self, attention):       
        
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
        
        # plt.show()

    
    def car_patch_pos(self, posture):
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.simulation_options["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.simulation_options["car size"][1]/2)
        
        return posture_new
    
    
    def on_press(self, event):
        
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
            
            # self.base_plot()
            self.show_attention(attention)
            self.fig.canvas.draw()