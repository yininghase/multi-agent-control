import os

import torch
import numpy as np
from torch_scatter import scatter
from argparse import ArgumentParser

from data_process import load_data, load_yaml 


def calculate_metrics(config): 
    
    assert os.path.exists(config["data folder"]), \
        f"The given folder of '{config['data folder']}' does not exist!"        
    
    data = load_data(num_vehicles = config["number of vehicles"], 
                     num_obstacles = config["num of obstacles"],
                     folders = config["data folder"],
                     load_all_simpler = True,
                     horizon = 1,
                     load_trajectory=True,
                     load_model_prediction=True)
    
    metrics= {"success to goal":{},
              "collision states": {},
              "collision times": {},
              "travel distance":{},
              "success to goal rate":{},
              "collision rate": {},
              "number of trajectories": {},
              "trajectory efficiency": {},
              }
    
    for key, value in data.items():
        
        len_batch = key[0]
        len_vehicle = key[1]
        len_obstacle = key[0]-key[1]
        
        states, _, _, trajectory_idx = value
        metrics["number of trajectories"].update({key:len(trajectory_idx)-1})
        
        states = states.reshape(-1, len_batch, 8)
        trajectory_idx = (trajectory_idx/len_batch).type(torch.long)
        
        trajectory = states[:,:len_vehicle,:2]
        
        travel_distance = torch.norm(torch.diff(trajectory, dim=0, prepend=trajectory[0,:,:][None,...]), p=2, dim=-1)
        travel_distance = torch.cumsum(travel_distance, dim=0)
        travel_distance = travel_distance[trajectory_idx[1:]-1]-travel_distance[trajectory_idx[:-1]]
        metrics["travel distance"].update({key:travel_distance})
        
        collision_states, vehicle_with_collision = calculate_collision_times(states, vehicle_size=config["car size"], 
                                                    len_batch=len_batch, len_vehicle=len_vehicle, len_obstacle=len_obstacle)
        
        metrics["collision states"].update({key:collision_states})
        
        if key == (1,1):
            
            collision_times = torch.zeros((len(trajectory_idx)-1,1))
            metrics["collision times"].update({key:collision_times})
            metrics["collision rate"].update({key:0})
        
        else:
            collision_states = collision_states.clone().type(torch.int)
            collision_states = torch.clip(torch.diff(collision_states, dim=0, prepend=torch.zeros((1,collision_states.shape[1]))), min=0, max=None)
            collision_times = torch.cumsum(collision_states, dim=0)
            
            collision_times = torch.cat((torch.zeros((1,collision_times.shape[1])),collision_times))
            collision_times = collision_times[trajectory_idx[1:]] - collision_times[trajectory_idx[:-1]]
            metrics["collision times"].update({key:collision_times})
            collision_rate = torch.sum(collision_times)/torch.sum(travel_distance)
            metrics["collision rate"].update({key:collision_rate})
        
        vehicle_with_collision = torch.cat((torch.zeros((1,vehicle_with_collision.shape[1]), dtype=torch.bool), 
                                            vehicle_with_collision))
        vehicle_with_collision = torch.cumsum(vehicle_with_collision, dim=0)
        vehicle_with_collision = (vehicle_with_collision[trajectory_idx[1:]] - vehicle_with_collision[trajectory_idx[:-1]]) > 0
        
        final_states = states[trajectory_idx[1:]-1,:len_vehicle,:]
        reach_goal = calculate_reach_goal(final_states, 
                                            position_tolerance = config["position tolerance"], 
                                            angle_tolerance = config["angle tolerance"])
        
        success_to_goal = (~vehicle_with_collision) * reach_goal
        
        metrics["success to goal"].update({key:success_to_goal})
        
        success_to_goal_rate = torch.sum(success_to_goal)/(len_vehicle*len(success_to_goal))
        metrics["success to goal rate"].update({key:success_to_goal_rate})
        
        trajectory_efficiency = calculate_trajectory_efficiency(travel_distance, 
                                    states[trajectory_idx[:-1],:len_vehicle,:2], 
                                    states[trajectory_idx[:-1],:len_vehicle,4:6], 
                                    success_to_goal)
        
        metrics["trajectory efficiency"].update({key:trajectory_efficiency})
                
        print(f"number of vehicle: {len_vehicle}, number of obstacle: {len_obstacle}")
        print(f"number of trajectories: {metrics['number of trajectories'][key]}")
        print(f"success to goal rate: {metrics['success to goal rate'][key]}")
        print(f"collision rate: {metrics['collision rate'][key]}")
        print(f"trajectory efficiency: {metrics['trajectory efficiency'][key]}")
        print()
        

def calculate_step_efficiency(config):
    
    assert os.path.exists(config["result parallel with edges"]) & os.path.exists(config["result sequential without edges"]), \
        f"The given folder of '{config['result parallel with edges']}' or '{config['result sequential without edges']}' does not exist!" 
    
    n = 10 # if the vehicle has not moved more then "stop tolerance" for n steps we end the simulation
    
    data_1 = load_data(num_vehicles = config["number of vehicles"], 
                     num_obstacles = config["num of obstacles"],
                     folders = config["result parallel with edges"],
                     load_all_simpler = True,
                     horizon = 1,
                     load_trajectory=True,
                     load_model_prediction=True)
    
    data_2 = load_data(num_vehicles = config["number of vehicles"], 
                     num_obstacles = config["num of obstacles"],
                     folders = config["result sequential without edges"],
                     load_all_simpler = True,
                     horizon = 1,
                     load_trajectory=True,
                     load_model_prediction=True)
    
    pos_stop_tol = config["stop tolerance"]
    ang_stop_tol = 2*config["stop tolerance"]
    
    step_effciency= {}
    
    for key, value_1 in data_1.items():
        
        len_batch = key[0]
        len_vehicle = key[1]
        len_obstacle = key[0]-key[1]
        
        states, _, _, trajectory_idx = value_1
        
        states = states.reshape(-1, len_batch, 8)
        trajectory_idx = (trajectory_idx/len_batch).type(torch.long)
        
        trajectory_1 = states[:,:len_vehicle,:3]
        
        collision_states, vehicle_with_collision = calculate_collision_times(states, vehicle_size=config["car size"], 
                                                    len_batch=len_batch, len_vehicle=len_vehicle, len_obstacle=len_obstacle)
        
        if key == (1,1):
            
            collision_times = torch.zeros((len(trajectory_idx)-1,1))
        
        else:
            collision_states = collision_states.clone().type(torch.int)
            collision_states = torch.clip(torch.diff(collision_states, dim=0, prepend=torch.zeros((1,collision_states.shape[1]))), min=0, max=None)
            collision_times = torch.cumsum(collision_states, dim=0)
            
            collision_times = torch.cat((torch.zeros((1,collision_times.shape[1])),collision_times))
            collision_times = collision_times[trajectory_idx[1:]] - collision_times[trajectory_idx[:-1]]
                    
        vehicle_with_collision = torch.cat((torch.zeros((1,vehicle_with_collision.shape[1]), dtype=torch.bool), 
                                            vehicle_with_collision))
        vehicle_with_collision = torch.cumsum(vehicle_with_collision, dim=0)
        vehicle_with_collision = (vehicle_with_collision[trajectory_idx[1:]] - vehicle_with_collision[trajectory_idx[:-1]]) > 0
        
        final_states = states[trajectory_idx[1:]-1,:len_vehicle,:]
        reach_goal = calculate_reach_goal(final_states, 
                                                    position_tolerance = config["position tolerance"], 
                                                    angle_tolerance = config["angle tolerance"])
        
        success_to_goal_1 = (~vehicle_with_collision) * reach_goal
        trajectory_idx_1 = trajectory_idx.clone()
        
        value_2 = data_2[key]
        states, _, _, trajectory_idx = value_2
        
        states = states.reshape(-1, len_batch, 8)
        trajectory_idx = (trajectory_idx/len_batch).type(torch.long)
        
        trajectory_2 = states[:,:len_vehicle,:3]
        
        collision_states, vehicle_with_collision = calculate_collision_times(states, vehicle_size=config["car size"], 
                                                    len_batch=len_batch, len_vehicle=len_vehicle, len_obstacle=len_obstacle)
        
        if key == (1,1):
            
            collision_times = torch.zeros((len(trajectory_idx)-1,1))
        
        else:
            collision_states = collision_states.clone().type(torch.int)
            collision_states = torch.clip(torch.diff(collision_states, dim=0, prepend=torch.zeros((1,collision_states.shape[1]))), min=0, max=None)
            collision_times = torch.cumsum(collision_states, dim=0)
            
            collision_times = torch.cat((torch.zeros((1,collision_times.shape[1])),collision_times))
            collision_times = collision_times[trajectory_idx[1:]] - collision_times[trajectory_idx[:-1]]
                    
        vehicle_with_collision = torch.cat((torch.zeros((1,vehicle_with_collision.shape[1]), dtype=torch.bool), 
                                            vehicle_with_collision))
        vehicle_with_collision = torch.cumsum(vehicle_with_collision, dim=0)
        vehicle_with_collision = (vehicle_with_collision[trajectory_idx[1:]] - vehicle_with_collision[trajectory_idx[:-1]]) > 0
        
        final_states = states[trajectory_idx[1:]-1,:len_vehicle,:]
        reach_goal = calculate_reach_goal(final_states, 
                                            position_tolerance = config["position tolerance"], 
                                            angle_tolerance = config["angle tolerance"])
        
        success_to_goal_2 = (~vehicle_with_collision) * reach_goal
        trajectory_idx_2 = trajectory_idx.clone()
        
        success_case = torch.all(success_to_goal_1, dim=-1) & torch.all(success_to_goal_2, dim=-1)
        success_case_idx = torch.where(success_case)[0]
        
        travel_step_1 = torch.diff(trajectory_1, dim=0, prepend=trajectory_1[0,:,:][None,...])
        travel_step_2 = torch.diff(trajectory_2, dim=0, prepend=trajectory_2[0,:,:][None,...])
        
        travel_step_n_1 = torch.zeros_like(travel_step_1)
        travel_step_n_2 = torch.zeros_like(travel_step_2)
        
        for i in range(n):
            travel_step_n_1 += torch.cat((travel_step_1[i:], torch.zeros((i, len_vehicle, travel_step_1.shape[-1]))), dim=0)
            travel_step_n_2 += torch.cat((travel_step_2[i:], torch.zeros((i, len_vehicle, travel_step_2.shape[-1]))), dim=0)
        
        early_stop_check_1 = (torch.norm(travel_step_n_1[:,:,:2], p=2, dim=-1)<pos_stop_tol) & (travel_step_n_1[:,:,2]<ang_stop_tol)
        early_stop_check_1 = torch.all(early_stop_check_1, dim=-1)
        early_stop_check_2 = (torch.norm(travel_step_n_2[:,:,:2], p=2, dim=-1)<pos_stop_tol) & (travel_step_n_2[:,:,2]<ang_stop_tol)
        
        early_stop_index_1 = torch.where(early_stop_check_1)[0]
        early_stop_index_2 = torch.stack(torch.where(early_stop_check_2), dim=1)
        
        early_stop_traj_index_1 = torch.bucketize(early_stop_index_1, trajectory_idx_1)-1
        early_stop_traj_index_2 = torch.bucketize(early_stop_index_2[:,0], trajectory_idx_2)-1
        
        early_stop_index_1 = scatter(early_stop_index_1, index=early_stop_traj_index_1, reduce="min")
        early_stop_index_1 = early_stop_index_1[early_stop_index_1!=0]
        
        temp = []
        
        for i in range(len_vehicle):
            idx = early_stop_traj_index_2[early_stop_index_2[:,1]==i]
            val = early_stop_index_2[:,0][early_stop_index_2[:,1]==i]
            idx = scatter(val, index=idx, reduce="min")
            idx = idx[idx!=0]
            idx = torch.stack((idx, i*torch.ones(len(idx), dtype=idx.dtype)), dim=1)
            temp.append(idx)
            
        early_stop_index_2 = torch.cat(temp, dim=0)
        
        early_stop_traj_index_1 = torch.bucketize(early_stop_index_1, trajectory_idx_1)-1
        early_stop_traj_index_2 = torch.bucketize(early_stop_index_2[:,0], trajectory_idx_2)-1
        
        valid_case_1 = torch.any(early_stop_traj_index_1[:,None] == success_case_idx[None,:], dim=-1)
        valid_case_2 = torch.any(early_stop_traj_index_2[:,None] == success_case_idx[None,:], dim=-1)
        
        early_stop_traj_index_1 = early_stop_traj_index_1[valid_case_1]
        early_stop_traj_index_2 = early_stop_traj_index_2[valid_case_2]
        early_stop_index_1 = early_stop_index_1[valid_case_1]
        early_stop_index_2 = early_stop_index_2[valid_case_2]
        
        total_step_1 = trajectory_idx_1[1:]-trajectory_idx_1[:-1]
        reduce_step = trajectory_idx_1[early_stop_traj_index_1+1]-trajectory_idx_1[early_stop_traj_index_1]-early_stop_index_1-n
        reduce_step = torch.clip(reduce_step, min=0)
        total_step_1[early_stop_traj_index_1] -= reduce_step
        total_step_1 = total_step_1[success_case_idx]
        
        total_step_2 = (trajectory_idx_2[1:]-trajectory_idx_2[:-1])*len_vehicle
        reduce_step = trajectory_idx_2[early_stop_traj_index_2+1]-trajectory_idx_2[early_stop_traj_index_2]-early_stop_index_2[:,0]-n
        reduce_step = torch.clip(reduce_step, min=0)
        
        for i in range(len_vehicle): 
            idx = early_stop_traj_index_2[early_stop_index_2[:,1]==i]    
            total_step_2[idx] -= reduce_step[early_stop_index_2[:,1]==i]
        total_step_2 = total_step_2[success_case_idx]
        
        eff = torch.sum(total_step_1)/torch.sum(total_step_2)
        step_effciency[key] = eff
        
        print(f"number of vehicle: {len_vehicle}, number of obstacle: {len_obstacle}")
        print(f"step efficiency: {step_effciency[key]}")
        print()
        

def calculate_collision_times(states, vehicle_size, len_batch, len_vehicle, len_obstacle):
    
    vehicle_with_collision = torch.zeros((len(states),len_vehicle))
    
    collisions = torch.empty((len(states),0), dtype=bool)
    
    if len_vehicle > 1:
        for i in range(len_vehicle-1):
            for j in range(i+1,len_vehicle):
                
                collisions_ij = check_collision_rectangular_rectangular(states[:,i,:], states[:,j,:], vehicle_size)
                vehicle_with_collision[:, i] += collisions_ij
                vehicle_with_collision[:, j] += collisions_ij
                collisions = torch.cat((collisions, collisions_ij[:,None]), dim=-1)
                
    
    if len_obstacle > 0:
        
        for i in range(len_vehicle):
            for j in range(len_vehicle, len_batch):
                
                collisions_ij = check_collision_rectangular_circle(states[:,i,:], states[:,j,:], vehicle_size)
                vehicle_with_collision[:, i] += collisions_ij
                collisions = torch.cat((collisions, collisions_ij[:,None]), dim=-1)
    
    return collisions, (vehicle_with_collision>0)


def check_collision_rectangular_circle(state_rect, state_cir, vehicle_size):
    
    assert len(state_rect) == len(state_cir) or len(state_cir) == 1, \
        "Mismatch of data length in collision check of one vehicle and one obstacle!"
    
    vehicle_corners_left_top = torch.tensor([vehicle_size[1]/2, vehicle_size[0]/2])
    
    inv_rotation = torch.empty((len(state_rect),2,2))
    
    inv_rotation[:,0,0] = torch.cos(state_rect[:,2])
    inv_rotation[:,0,1] = torch.sin(state_rect[:,2])
    inv_rotation[:,1,0] = -torch.sin(state_rect[:,2])
    inv_rotation[:,1,1] = torch.cos(state_rect[:,2])
    
    vect_rect_to_cir = (inv_rotation @ (state_cir[:,4:6,None]-state_rect[:,:2,None])).squeeze(-1)
    
    min_dist = torch.norm(torch.clip((torch.abs(vect_rect_to_cir) - vehicle_corners_left_top), min=0, max=None),p=2,dim=-1)
    
    collision = (min_dist - state_cir[:,6]) <= 0    
    
    return collision

def check_collision_rectangular_rectangular(state_1, state_2, vehicle_size):
    
    assert len(state_1) == len(state_2), \
        "Mismatch of data length in collision check of two vehicles!"
    
    not_collision = torch.zeros(len(state_1), dtype=bool)
    
    proj_vect = []
    
    vehicle_corners = torch.tensor([[vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, -vehicle_size[0]/2],
                                    [vehicle_size[1]/2, -vehicle_size[0]/2],
                                    ])
    
    rotation_1 = torch.empty((len(state_1),2,2))
    
    rotation_1[:,0,0] = torch.cos(state_1[:,2])
    rotation_1[:,0,1] = -torch.sin(state_1[:,2])
    rotation_1[:,1,0] = torch.sin(state_1[:,2])
    rotation_1[:,1,1] = torch.cos(state_1[:,2])
    
    vehicle_corners_1 = (rotation_1[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_1 = vehicle_corners_1 + state_1[:,None,:2]
    
    proj_vect.append(rotation_1[:,:,0])
    proj_vect.append(rotation_1[:,:,1])
    
    rotation_2 = torch.empty((len(state_2),2,2))
    
    rotation_2[:,0,0] = torch.cos(state_2[:,2])
    rotation_2[:,0,1] = -torch.sin(state_2[:,2])
    rotation_2[:,1,0] = torch.sin(state_2[:,2])
    rotation_2[:,1,1] = torch.cos(state_2[:,2])
    
    vehicle_corners_2 = (rotation_2[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_2 = vehicle_corners_2+ state_2[:,None,:2]
    
    proj_vect.append(rotation_2[:,:,0])
    proj_vect.append(rotation_2[:,:,1])
    
    for i in range(4):
        
        if torch.all(not_collision):
            break
            
        proj_vehicle_1 = (vehicle_corners_1[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        proj_vehicle_2 = (vehicle_corners_2[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        
        not_collision[(not_collision==False)] |= (torch.amax(proj_vehicle_1, dim=-1)<torch.amin(proj_vehicle_2, dim=-1)) | \
                                                 (torch.amax(proj_vehicle_2, dim=-1)<torch.amin(proj_vehicle_1, dim=-1))
    
    return ~not_collision


def calculate_reach_goal(final_states, position_tolerance, angle_tolerance):
    
    pos_diff = torch.norm(final_states[:,:,:2]-final_states[:,:,4:6],p=2,dim=-1)
    
    angle_diff_1 = (final_states[:,:,2] - final_states[:,:,6])[...,None]%(2*np.pi)
    angle_diff_2 = 2*np.pi - angle_diff_1
    angle_diff = torch.amin(torch.concat((angle_diff_1,angle_diff_2), dim=-1), dim=-1)
    
    reach_goal = (pos_diff<=position_tolerance) & (angle_diff<=angle_tolerance)   
    
    return reach_goal


def calculate_trajectory_efficiency(trajectory_distance, start, goal, success_index):
        
    start_goal_distance = torch.norm(start-goal, p=2, dim=-1)[success_index]
    trajectory_distance = trajectory_distance[success_index]
    trajectory_efficiency = torch.sum(start_goal_distance)/torch.sum(trajectory_distance)
    
    return trajectory_efficiency


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/calculate_metrics.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    if not isinstance(config["data folder"], str):
        data_folders = config["data folder"].copy()
        
        for data_folder in data_folders:
            print(f"Currently evaluating {data_folder}: ")
            config["data folder"] = data_folder
            calculate_metrics(config)
            print("*"*10)
        
    else:
        print(f"Currently evaluating {config['data folder']}: ")
        calculate_metrics(config)
        print("*"*10)
    
    
    if not isinstance(config["result parallel with edges"], str):
        parallel_with_edges = config["result parallel with edges"].copy()
        sequential_without_edges = config["result sequential without edges"].copy()
        
        assert len(parallel_with_edges) == len(sequential_without_edges), \
            '"result parallel with edges" and "result sequential without edges" should match!'
        
        for i in range(len(parallel_with_edges)):
            parallel_with_edges_i = parallel_with_edges[i]
            sequential_without_edges_i = sequential_without_edges[i]
            
            assert os.path.basename(parallel_with_edges_i)+'_NoVehicleEdges' == os.path.basename(sequential_without_edges_i), \
            '"result parallel with edges" and "result sequential without edges" should match!'
            
            config["result parallel with edges"] = parallel_with_edges_i
            config["result sequential without edges"] = sequential_without_edges_i
            print(f"Currently calculating step efficiency of {parallel_with_edges_i}: ")
            calculate_step_efficiency(config)
            print("*"*10)
    
    else:
        parallel_with_edges = config["result parallel with edges"]
        sequential_without_edges = config["result sequential without edges"]
        assert os.path.basename(parallel_with_edges)+'_NoVehicleEdges' == os.path.basename(sequential_without_edges), \
            '"result parallel with edges" and "result sequential without edges" should match!'
        print(f"Currently calculating step efficiency of {parallel_with_edges}: ")
        calculate_step_efficiency(config)
        print("*"*10)