import os

import torch
import numpy as np
import random
import yaml

from numpy import pi

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def get_vehicles_more_collision(data_length, num_vehicles, position_range=25, collision_range = 15):
    '''
    This function generate the start and goal of the vehicles with higher probability to cause a crash 
    if the vehicles just drive directly towards the goal.
    '''
    
    assert num_vehicles > 1,\
    "The function get_vehicles_more_collision only work for the problem with more than 1 vehicle!"
    
    if data_length == 0:
        return np.empty((0,num_vehicles,7))
    
    low_range_pos = np.array([-collision_range, -collision_range])
    high_range_pos = np.array([collision_range, collision_range])
    mean_vel = 2.5
    variance_vel = 5
    low_range_shift = -position_range+collision_range
    high_range_shift = position_range-collision_range
    
    vehicle_index = np.ones(data_length, dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        min_dist = np.ones((data_length, 1)) * np.inf
        pos[vehicle_index] = np.random.uniform(low=low_range_pos-3*failed_time//100, 
                                               high=high_range_pos+3*failed_time//100, 
                                               size=(np.sum(vehicle_index), num_vehicles, 2))
        
        for i in range(num_vehicles-1):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)[:,None]
                min_dist = np.amin(np.concatenate((min_dist, dist),axis=-1), axis=-1)[:,None]
        
        vehicle_index = min_dist.squeeze(-1)<5
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
    
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    vel = np.random.normal(loc=mean_vel, scale=variance_vel, size=(data_length, num_vehicles, 1))
    starts = np.concatenate((pos,angle,vel), axis=-1)
    
    shift_center = np.random.uniform(low=low_range_shift, high=high_range_shift, size=(data_length,2))   
    
    centers = np.mean(starts[:,:,:2], axis=1, keepdims=True)
    v = centers-starts[:,:,:2]
    v = v/np.linalg.norm(v,axis=-1,keepdims=True)
    h = v[:,:,[1,0]]
    h[:,:,0] *= -1 
        
    vehicle_index = np.ones(data_length, dtype=bool)
    targets = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        min_dist = np.ones((data_length, 1)) * np.inf
    
        shifts_v = np.random.uniform(low=0, high=20+3*failed_time//100, size=(np.sum(vehicle_index), num_vehicles, 1))
        shifts_h = np.random.normal(loc=0, scale=3, size=(np.sum(vehicle_index), num_vehicles, 1))
        
        targets[vehicle_index] = centers[vehicle_index]+shifts_v*v[vehicle_index]+shifts_h*h[vehicle_index]
        
        for i in range(num_vehicles-1):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(targets[:,i,:2]-targets[:,j,:2],ord=2,axis=-1)[:,None]
                min_dist = np.amin(np.concatenate((min_dist, dist),axis=-1), axis=-1)[:,None]
        
        vehicle_index = min_dist.squeeze(-1)<5
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
        
    
    targets_angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    targets = np.concatenate((targets, targets_angle), axis=-1)
    starts[...,:2] += shift_center[:,None,:]
    targets[...,:2] += shift_center[:,None,:]
    
    return np.append(starts, targets, axis=2)


def get_vehicles_normal(data_length, num_vehicles, position_range=25):
    '''
    This function generate the normal case of start and goal of the vehicles with guarantee that
    the goal of the vehicles are far away enough from each other (more than 5).
    '''
    
    if data_length == 0:
        return np.empty((0,num_vehicles,7))
    
    low_range_pos = [-position_range, -position_range, -np.pi]
    high_range_pos = [position_range, position_range, np.pi]
    mean_vel = 2.5
    variance_vel = 5
    
    pos = np.random.uniform(low=low_range_pos, high=high_range_pos, size=(data_length, num_vehicles, 3))
    vel = np.random.normal(loc=mean_vel, scale=variance_vel, size=(data_length, num_vehicles, 1))
    starts = np.concatenate((pos,vel), axis=-1)
    
    targets = np.random.uniform(low=low_range_pos, high=high_range_pos, size=(data_length, num_vehicles, 3))
    
    if num_vehicles > 1:
    
        min_dist = np.empty((data_length, 0))
        
        for i in range(num_vehicles-1):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(targets[:,i,:2]-targets[:,j,:2],ord=2, axis=-1)[:,None]
                min_dist = np.amin(np.concatenate((min_dist, dist), axis=-1), axis=-1)[:,None]
        
        min_dist = min_dist.squeeze(-1)
        valid = min_dist<=5
        factor = (5/min_dist)*np.random.uniform(low=1, high=1.2, size=(len(min_dist)))
        
        centers = np.mean(targets[:,:,:2], axis=1, keepdims=True)
        v = targets[:,:,:2] - centers
        targets[valid,:,:2] = centers[valid,...] + v[valid,:,:2]*factor[valid,None,None]
    
    return np.append(starts, targets, axis=2)


def get_vehicles_parking(data_length, num_vehicles, position_range=25):
    '''
    This function generate the start and goal of the vehicles with guarantee that
    at least one of the vehicles is in parking mode (its start is close to its goal).
    '''
    
    if data_length == 0:
        return np.empty((0,num_vehicles,7))
    
    low_range_pos = [-position_range, -position_range, -np.pi]
    high_range_pos = [position_range, position_range, np.pi]
    
    targets = np.random.uniform(low=low_range_pos, high=high_range_pos, size=(data_length, num_vehicles, 3))
    
    if num_vehicles > 1:
        min_dist = np.ones((data_length, 1)) * np.inf
        
        for i in range(num_vehicles-1):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(targets[:,i,:2]-targets[:,j,:2],ord=2,axis=-1)[:,None]
                min_dist = np.amin(np.concatenate((min_dist, dist),axis=-1), axis=-1)[:,None]
        min_dist = min_dist.squeeze(-1)
        valid = min_dist<=5
        factor = (5/min_dist)*np.random.uniform(low=1, high=1.2, size=(len(min_dist)))
        
        centers = np.mean(targets[:,:,:2], axis=1, keepdims=True)
        v = targets[:,:,:2] - centers
        targets[valid,:,:2] = centers[valid,...] + v[valid,:,:2]*factor[valid,None,None]
    
    
    ref_vel = np.random.normal(loc=1.5, scale=5, size=(data_length))
    min_dist = np.random.uniform(low=0, high=7, size=(data_length))
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    vel = np.random.normal(loc=2.5, scale=5, size=(data_length, num_vehicles, 1))
    pos = np.ones((data_length, num_vehicles, 2))
    
    rel_angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles))
    rel_dist = np.random.uniform(low=0, high=20, size=(data_length, num_vehicles))
    arg_min = np.argmin(rel_dist, axis=-1)
    rel_dist[range(data_length),arg_min] = min_dist 
    pos[range(data_length),:,0]=targets[:,:,0]+rel_dist*np.cos(rel_angle)
    pos[range(data_length),:,1]=targets[:,:,1]+rel_dist*np.sin(rel_angle)
    
    vel[range(data_length),arg_min,0] = ref_vel
    starts = np.concatenate((pos,angle,vel), axis=-1)
    
    return np.append(starts, targets, axis=2)


def get_obstacles_one(starts, targets, num_obstacles):
    '''
    This function generate the obstacles with guarantee that
    the obstacles are near the route of the vehicles as well as
    not too close to their start and goal.
    '''
    if num_obstacles == 0:
        return np.empty((0,3))
    
    # obstacle should keep far from start and goal of all the vehicles to avoid the vehicles get stack
    check_point = np.concatenate((starts,targets),axis=0) 
    
    v = targets - starts
    d = np.linalg.norm(v, ord=2,axis=-1)
    
    v = v/d[:,None]
    h = v[:,[1,0]]
    h[:,0] *= -1
    
    obstacles = np.empty((0,3))
    
    while(num_obstacles!= 0):
    
        h_shift1 = np.random.uniform(low=-7, high=0, size=(num_obstacles, len(d)))
        h_shift2 = np.random.uniform(low=0, high=7, size=(num_obstacles, len(d)))
        h_shift = np.concatenate((h_shift1,h_shift2),axis=0)
        v_shift = np.random.uniform(low=[-7]*len(d), high=(d+7).tolist(), size=(2*num_obstacles, len(d)))
        r = np.random.uniform(low=1, high=3, size=(2*num_obstacles*len(d),1))
        
        pos = (starts + v*v_shift[:,:,None] + h*h_shift[:,:,None]).reshape(-1,2)
        
        dist = np.linalg.norm(pos[:,None,:]-check_point, ord=2, axis=-1)
        min_dist = np.amin(dist, axis=-1)
        valid = (min_dist - r.flatten() -5) > 0
        
        candidates = np.concatenate((pos,r),axis=-1)[valid]
        
        if len(candidates) <= 0:
            continue
        
        n = min(len(candidates), num_obstacles)
        
        chosen = np.random.choice(range(len(candidates)), n, replace=False)           
        obstacles = np.concatenate((obstacles, candidates[chosen]), axis=0)
        
        num_obstacles -= n
    
    return obstacles


def get_obstacles_batch(starts, targets, num_obstacles):
    '''
    This function generate the obstacles with guarantee that
    the obstacles are near the route of the vehicles as well as
    not too close to their start and goal.
    '''
    
    assert starts.shape == targets.shape, \
        "Error input of starts or targets!"
    
    num_batches = starts.shape[0]
    num_vehicles = starts.shape[1]
    
    if num_obstacles == 0:
        return np.empty((num_batches, 0, 3))
    
    # obstacle should keep far from start and goal of all the vehicles to avoid the vehicles get stack
    
    
    # (batch, vehicle, 2)
    v = targets - starts
    # (batch, vehicle)
    d = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
    
    v = v/d
    h = v[...,[1,0]]
    h[...,0] *= -1
    
    obstacles_ = np.empty((num_batches, num_vehicles, num_obstacles, 3))
    obstacles_index = np.ones((num_batches, num_vehicles, num_obstacles), dtype=bool)
    v_shift_high = np.tile(d, (1,1, num_obstacles))+7
    starts_ = np.tile(starts[:,:,None,:],(1,1,num_obstacles,1))
    v = np.tile(v[:,:,None,:],(1,1,num_obstacles,1))
    h = np.tile(h[:,:,None,:],(1,1,num_obstacles,1))
    
    check_point = np.concatenate((starts,targets),axis=-2) 
    
    failed_time = 0
    last_failed = np.sum(obstacles_index)
    
    while(last_failed > 0):
        
        h_shift = np.random.uniform(low=0, high=7+3*failed_time//100, size=np.sum(obstacles_index))
        v_shift = np.random.uniform(low=[-7-3*failed_time//100]*np.sum(obstacles_index), 
                                    high=(v_shift_high[obstacles_index]+3*failed_time//100).tolist())
        v_flip = np.random.randint(low=0, high=2, size=len(v_shift))*2-1
        v_shift = v_shift*v_flip
        r = np.random.uniform(low=1, high=3, size=np.sum(obstacles_index))
        
        obstacles_[obstacles_index,:2] = (starts_[obstacles_index] + v[obstacles_index]*v_shift[:,None] 
                                         + h[obstacles_index]*h_shift[:,None]).reshape(-1,2)
        obstacles_[obstacles_index,2] = r
        
        dist = np.linalg.norm(obstacles_[:,:,:,None,:2]-check_point[:,None,None,:,:], ord=2, axis=-1)
        min_dist = np.amin(dist, axis=-1)
        obstacles_index = (min_dist - obstacles_[:,:,:,2] - 5) <= 0
        
        if last_failed <= np.sum(obstacles_index):
            failed_time += 1
            
        last_failed = np.sum(obstacles_index)
    
    obstacles_ = obstacles_.reshape((num_batches, -1, 3))
    
    chosen = np.random.rand(num_batches, num_vehicles*num_obstacles)
    chosen = np.argsort(chosen, axis=-1)[:,:num_obstacles]
    obstacles = obstacles_[np.arange(len(chosen))[:,None].repeat(num_obstacles, axis=-1), chosen]
        
    return obstacles


def get_problem(num_vehicles, num_obstacles, collision = True, parking = False, data_length = 1, mode = "generate train trajectory"):
    ''' The function to get the problem with num_vehicles vehicles and num_obstacles obstacles'''
    
    assert not(collision and parking), \
        "Can not implement collision mode and parking mode at the same time!"
    
    if mode in ["generate train trajectory", "inference"]:
        
        assert data_length == 1, \
               "Return more than 1 problem can only be used under generate test problem!"
    
        if collision and (num_vehicles > 1):
            vehicles = get_vehicles_more_collision(data_length, num_vehicles)[0]
            
        
        elif parking:
            vehicles = get_vehicles_parking(data_length, num_vehicles)[0]
        
        else:
            vehicles = get_vehicles_normal(data_length, num_vehicles)[0]
            
        
        starts = vehicles[:,:4]
        starts[:,-1] = 0
        targets = vehicles[:,4:]
        
        obstacles = get_obstacles_one(starts[:,:2], targets[:,:2], num_obstacles)
        
        return starts, targets, obstacles
    
    elif mode in ["generate test problem"]:
        
        if collision and (num_vehicles > 1):
            vehicles = get_vehicles_more_collision(data_length, num_vehicles)
            
        
        elif parking:
            vehicles = get_vehicles_parking(data_length, num_vehicles)
        
        else:
            vehicles = get_vehicles_normal(data_length, num_vehicles)
        
        obstacles = get_obstacles_batch(vehicles[...,:2], vehicles[...,4:6], num_obstacles)
        
        if mode == "generate test data":
            vehicles[:,:,3] = 0
            vehicles = np.concatenate((vehicles, 
                                       np.zeros((vehicles.shape[0], vehicles.shape[1],1))), axis=-1)
            obstacles = np.concatenate((np.zeros((obstacles.shape[0], obstacles.shape[1],4)),
                                        obstacles,
                                        np.ones((obstacles.shape[0], obstacles.shape[1],1)),
                                        ), axis=-1)

        return vehicles, obstacles
        
    else:
        raise NotImplementedError("Unknown mode of get problem!")


def load_data(num_vehicles, num_obstacles, load_all_simpler=True, folders="./data/data_generation", 
              horizon=0, load_trajectory=False, load_model_prediction=False):
    '''load data from folder'''
    data = {}
    
    if isinstance(folders, str):
        folders = [folders]
    else:
        assert isinstance(folders, list), \
            "invalid input of data folders, should be string of list or string"
    
    if load_all_simpler:
        # load all simpler case with less vehicles and obstacles
        for i in range(1, num_vehicles+1):
            for j in range(num_obstacles+1):
                for folder in folders:
                    X_data_path = os.path.join(folder, f"X_data_vehicle={i}_obstalce={j}.pt")
                    
                    if load_model_prediction:
                        y_data_path = os.path.join(folder, f"y_model_data_vehicle={i}_obstalce={j}.pt")  
                    else:
                        y_data_path = os.path.join(folder, f"y_GT_data_vehicle={i}_obstalce={j}.pt")
                                            
                    batches_data_path = os.path.join(folder, f"batches_data_vehicle={i}_obstalce={j}.pt")
                    
                    if load_trajectory:
                        trajectory_data_path = os.path.join(folder, f"trajectory_data_vehicle={i}_obstalce={j}.pt")
                        if not os.path.exists(trajectory_data_path):
                            continue
    
                    if os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path):
                        
                        new_X_data = torch.load(X_data_path).type(torch.float32)
                        new_y_data = torch.load(y_data_path).type(torch.float32)
                        new_batches_data = torch.load(batches_data_path).type(torch.int)
                        
                        if isinstance(horizon, int) and horizon>0 and 2*horizon<new_y_data.shape[-1]:
                            new_y_data = new_y_data[:,:2*horizon]
                        
                        if load_trajectory:
                            new_trajectory_data = torch.load(trajectory_data_path).type(torch.int)
                        
                        if (i+j,i) in data.keys():
                            
                            if load_trajectory:
                                X_data, y_data, batches_data, trajectory_data = data[(i+j,i)]
                            else:
                                X_data, y_data, batches_data = data[(i+j,i)]
                            
                            X_data = torch.cat((X_data, new_X_data))
                            y_data = torch.cat((y_data, new_y_data))
                            batches_data = torch.cat((batches_data, new_batches_data))
                            
                            if load_trajectory:
                                trajectory_data = torch.cat((trajectory_data, new_trajectory_data[1:]+trajectory_data[-1]))
                                data[(i+j,i)] = [X_data, y_data, batches_data, trajectory_data]
                            
                            else:
                                data[(i+j,i)] = [X_data, y_data, batches_data]
                            
                        else:
                            
                            if load_trajectory:
                                data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data, new_trajectory_data]
                            else:
                                data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data]
                        
    else:
        # load only the case with of given number of vehicles and obstacles
        
        i = num_vehicles
        j = num_obstacles
        
        for folder in folders:
            X_data_path = os.path.join(folder, f"X_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt")
            
            if load_model_prediction:
                y_data_path = os.path.join(folder, f"y_model_data_vehicle={i}_obstalce={j}.pt")  
            else:
                y_data_path = os.path.join(folder, f"y_GT_data_vehicle={i}_obstalce={j}.pt")
                
            batches_data_path = os.path.join(folder, f"batches_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt")
            
            if load_trajectory:
                trajectory_data_path = os.path.join(folder, f"trajectory_data_vehicle={i}_obstalce={j}.pt")
                if not os.path.exists(trajectory_data_path):
                    continue
            
            if not (os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path)):
                continue
            
            new_X_data = torch.load(X_data_path).type(torch.float32)
            new_y_data = torch.load(y_data_path).type(torch.float32)
            new_batches_data = torch.load(batches_data_path).type(torch.int)
            
            if isinstance(horizon, int) and horizon>0 and 2*horizon<new_y_data.shape[-1]:
                new_y_data = new_y_data[:,:2*horizon]
            
            if load_trajectory:
                new_trajectory_data = torch.load(trajectory_data_path).type(torch.int)
            
            
            if os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path):
                if (i+j,i) in data.keys():
                    
                    if load_trajectory:
                        X_data, y_data, batches_data, trajectory_data = data[(i+j,i)]
                    else:
                        X_data, y_data, batches_data = data[(i+j,i)]
                    
                    X_data = torch.cat((X_data, new_X_data))
                    y_data = torch.cat((y_data, new_y_data))
                    batches_data = torch.cat((batches_data, new_batches_data))
                    
                    if load_trajectory:
                        trajectory_data = torch.cat((trajectory_data, new_trajectory_data[1:]+trajectory_data[-1]))
                        data[(i+j,i)] = [X_data, y_data, batches_data, trajectory_data]
                    
                    else:
                        data[(i+j,i)] = [X_data, y_data, batches_data]
                    
                else:
                    
                    if load_trajectory:
                        data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data, new_trajectory_data]
                    else:
                        data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data]

        assert len(data[(i+j,i)][-1]) > 0,\
                f"The data with {num_vehicles} vehicles and {num_obstacles} obstacles does not exist!"
        
        
    return data


def load_test_data(num_vehicles, num_obstacles, load_all_simpler=True, folders="./data/test_dataset", lim_length=None):
    
    if isinstance(folders, str):
        folders = [folders]
    else:
        assert isinstance(folders, list), \
            "invalid input of data folders, should be string of list or string"
    
    data = []
    
    if load_all_simpler:
        # load all simpler case with less vehicles and obstacles
        for i in range(1, num_vehicles+1):
            for j in range(num_obstacles+1):
                
                test_data = torch.empty((0,i+j,8))
                
                for folder in folders:
                    
                    test_data_path = os.path.join(folder, f"test_data_vehicle={i}_obstalce={j}.pt")
    
                    if os.path.exists(test_data_path):
                        test_data = torch.cat((test_data, torch.load(test_data_path).type(torch.float32)))
                
                if isinstance(lim_length,int) and lim_length<len(test_data):
                    test_data = test_data[:lim_length]    
                
                starts = test_data[:,:i,:4]
                targets = test_data[:,:i, 4:7]
                obstacles = test_data[:,i:i+j,4:7]
                problem_mark = torch.tensor([i, j], dtype=int).repeat((len(test_data),1))
                
                data.extend(list(zip(starts,targets,obstacles,problem_mark)))            
                        
    else:
        # load only the case with of given number of vehicles and obstacles
        
        i = num_vehicles
        j = num_obstacles
        
        test_data = torch.empty((0,i+j,8))
        
        for folder in folders:
            
            test_data_path = os.path.join(folder, f"test_data_vehicle={i}_obstalce={j}.pt")

            if os.path.exists(test_data_path):
                test_data = torch.cat((test_data, torch.load(test_data_path).type(torch.float32)))
        
               
        if isinstance(lim_length,int) and lim_length<len(test_data):
            test_data = test_data[:lim_length]
        
        starts = test_data[:,:i,:4]
        targets = test_data[:,:i, 4:7]
        obstacles = test_data[:,i:i+j,4:7]
        
        problem_mark = torch.tensor([i, j], dtype=int).repeat((len(test_data),1))
                
        data.extend(list(zip(starts,targets,obstacles,problem_mark)))


        assert len(data) > 0,\
                f"The data with {num_vehicles} vehicles and {num_obstacles} obstacles does not exist!"
        
        
    return data


# need to be changed
def change_to_relative_frame(data, num_vehicles, num_obstacles):
    ''' The function to change the goal of the vehicle and the obstacle position to the local frame of vehicle'''
    assert num_vehicles == 1, "Only one vehicle mode can use relative frame mode!"
    
    data_reshape = data.reshape(-1, num_vehicles+num_obstacles, 8).type(torch.float32)
    vehicles = data_reshape[:,0,:]
    obstacles = data_reshape[:,1:,:] 
    
    rotation_matrices = torch.zeros((len(data_reshape), 2, 2))
    rotation_matrices[:, 0, 0] = torch.cos(vehicles[:, 2])
    rotation_matrices[:, 0, 1] = torch.sin(vehicles[:, 2])
    rotation_matrices[:, 1, 0] = -torch.sin(vehicles[:, 2])
    rotation_matrices[:, 1, 1] = torch.cos(vehicles[:, 2])
    
    relative_points = (vehicles[:, 4:7]-vehicles[:, :3])
    relative_points[:,:2] = torch.matmul(rotation_matrices, relative_points[:,:2,None]).squeeze(-1)
    vehicles[:, 4:7] = relative_points
    
    if num_obstacles>0:
    
        relative_points = (obstacles[:, :, 4:6]-vehicles[:, None, :2])
        relative_points = torch.matmul(rotation_matrices[:, None, :,:], relative_points[:,:,:,None]).squeeze(-1)
        obstacles[:, :, 4:6] = relative_points
        
    data_reshape[:,0,:] = vehicles
    data_reshape[:,1:,:] = obstacles
    data = data_reshape.reshape(-1,8)

    return data


def split_train_valid(data):
    '''This function is to split the training data and validation data'''
    train_data = {}
    valid_data = {}

    
    for key, values in data.items():
        
        X, y, batches =  values
        
        X = X.reshape(len(batches), key[0], -1)
        y = y.reshape(len(batches), key[1], -1)
        y_last_shape = y.shape[-1]
        
        train_index, valid_index = train_test_split(np.arange(len(batches)), test_size=0.2)
        
        train_X = X[train_index].reshape(len(train_index)*key[0], -1)
        train_y = y[train_index].reshape(len(train_index)*key[1], -1)
        train_batches = batches[train_index]
        
        valid_X = X[valid_index].reshape(len(valid_index)*key[0], -1)
        valid_y = y[valid_index].reshape(len(valid_index)*key[1], -1)
        valid_batches = batches[valid_index]
        
        assert train_X.shape[1] == 8 and valid_X.shape[1] == 8 and \
               train_y.shape[1] == y_last_shape and \
               valid_y.shape[1] == y_last_shape, \
                "Error in train and valid data split!"
        
        train_data[key] = [train_X, train_y, train_batches]
        valid_data[key] = [valid_X, valid_y, valid_batches]
        
    return train_data, valid_data
        

def get_angle_diff(angle_1, angle_2, mode="directed_vector"):
    
    if mode == "directed_vector":
        angle_diff_1 = (angle_1-angle_2)[...,None]%(2*np.pi)
        angle_diff_2 = 2*np.pi - angle_diff_1
        angle_diff = np.amin(np.concatenate((angle_diff_1,angle_diff_2), axis=-1), axis=-1)
        
    elif mode =="undirected_vector":
        angle_diff_1 = (angle_1-angle_2)[...,None]%(2*np.pi)
        angle_diff_2 = 2*np.pi - angle_diff_1
        angle_diff_3 = (angle_1+angle_2)[...,None]%(2*np.pi)
        angle_diff_4 = 2*np.pi - angle_diff_3
        angle_diff = np.amin(np.concatenate((angle_diff_1,angle_diff_2,angle_diff_3,angle_diff_4), axis=-1), axis=-1)
    
    return angle_diff


class GNN_Dataset(Dataset):
    def __init__(self, data, use_relative_frame=False, augmentation=True, sample_each_case=None):
        
        self.augmentation = augmentation
        self.data = []
        
        if use_relative_frame:
            keys = np.array(list(data.keys()))
            assert np.all(keys[:,1] == 1), \
            "relative frame can only be used in 1-vehicle case!"
            self.augmentation = False
        
        for key, value in data.items():
            
            len_batch, len_vehicle = key
            X, y, batch = value
            
            assert len(X) == sum(batch[:,0])
            assert len(y) == sum(batch[:,1])
            
            N = len(batch)
            
            if use_relative_frame:
                X = change_to_relative_frame(X, len_vehicle, len_batch-len_vehicle)
            
            X = X.reshape(N, len_batch, 8)
            y = y.reshape(N, len_vehicle, -1)
            batch = batch.type(torch.int)
            
            
            obstacle_random_angle = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=(N, len_batch-len_vehicle)))
            if not use_relative_frame:
                X[:,len_vehicle:len_batch,:2] = X[:,len_vehicle:len_batch,4:6]
                X[:,len_vehicle:len_batch,2] = obstacle_random_angle
                  
            X[:,len_vehicle:len_batch,-1] = X[:,len_vehicle:len_batch,-2]
            X[:,len_vehicle:len_batch,-2] = obstacle_random_angle
            
            X[:,:,2] = (X[:,:,2]+pi)%(2*pi)-pi
            X[:,:,-2] = (X[:,:,-2]+pi)%(2*pi)-pi
            
            if isinstance(sample_each_case, int) and sample_each_case>0:
                samples = list(zip(X,y,batch))
                random.shuffle(samples)
                self.data.extend(samples[:min(sample_each_case, len(samples))])
                # self.data.append(samples[15])
            else:
                self.data.extend(list(zip(X,y,batch)))
        
        random.shuffle(self.data)
            
                  
    def transform_coordinate_one(self, X):
        
        coord_min = torch.amin(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_max = torch.amax(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_center = (coord_max+coord_min)/2
        
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        R = torch.tensor([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
        
        X[:,:2] = (R@((X[:,:2]-coord_center)[:,:,None])).squeeze(-1)
        X[:,4:6] = (R@((X[:,4:6]-coord_center)[:,:,None])).squeeze(-1)
        
        X[:,2] = (X[:,2] + theta + np.pi)%(2*np.pi) - np.pi
        X[:,6] = (X[:,6] + theta + np.pi)%(2*np.pi) - np.pi
        
        coord_min = torch.amin(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_max = torch.amax(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        
        t_x = np.random.uniform(low=min(-10, -25-coord_min[0]), high=max(10, 25-coord_max[0]))
        t_y = np.random.uniform(low=min(-10, -25-coord_min[1]), high=max(10, 25-coord_max[1]))
        t = torch.tensor([t_x, t_y], dtype=torch.float32)
        
        X[:,:2] = X[:,:2] + t
        X[:,4:6] = X[:,4:6] + t
            
        return X
    
    def transform_coordinate_batch(self, X):
        
        coord_min = torch.amin(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_max = torch.amax(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_center = (coord_max+coord_min)/2
        
        theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=(len(X)))).type(torch.float32)
        r11 = torch.cos(theta)
        r12 = -torch.sin(theta)
        r21 = torch.sin(theta)
        r22 = torch.cos(theta)
        
        R = torch.empty((len(X),1,2,2), dtype=torch.float32)
        
        R[:,0,0,0] = r11
        R[:,0,0,1] = r12
        R[:,0,1,0] = r21
        R[:,0,1,1] = r22
        
        X[:,:,:2] = (R@((X[:,:,:2]-coord_center[:,None,:2])[:,:,:,None])).squeeze(-1)
        X[:,:,4:6] = (R@((X[:,:,4:6]-coord_center[:,None,:2])[:,:,:,None])).squeeze(-1)
        
        X[:,:,2] = (X[:,:,2] + theta[:,None] + np.pi)%(2*np.pi) - np.pi
        X[:,:,6] = (X[:,:,6] + theta[:,None] + np.pi)%(2*np.pi) - np.pi
        
        coord_min = torch.amin(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_max = torch.amax(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        
        t_x = torch.from_numpy(np.random.uniform(low=torch.clip(-25-coord_min[:,0], max=-10), 
                                                    high=torch.clip(25-coord_max[:,0], min=10), 
                                                    )).type(torch.float32)
        t_y = torch.from_numpy(np.random.uniform(low=torch.clip(-25-coord_min[:,1], max=-10), 
                                                    high=torch.clip(25-coord_max[:,1], min=10), 
                                                    )).type(torch.float32)
        t = torch.empty((len(X),1,2), dtype=torch.float32)
        t[:,0,0] = t_x
        t[:,0,1] = t_y
        
        X[:,:,:2] = X[:,:,:2] + t
        X[:,:,4:6] = X[:,:,4:6] + t
            
        return X
           
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        
        if self.augmentation:
            
            if isinstance(index, int):
                X,y,batch = self.data[index]
                X = self.transform_coordinate_one(X)                
                return (X,y,batch)
            
            elif isinstance(index, slice):
            
                X,y,batches = zip(*self.data[index])
                X = torch.cat(X)
                y = torch.cat(y)
                batches = torch.stack(batches)
                batches_offset_X = torch.cumsum(batches[:,0],dim=0)[:-1]
                batches_offset_X = torch.cat((torch.tensor([0]), batches_offset_X))
                batches_offset_y = torch.cumsum(batches[:,1],dim=0)[:-1]
                batches_offset_y = torch.cat((torch.tensor([0]), batches_offset_y))
                        
                data_augmented = []
                
                for batch in torch.unique(batches, dim=0):
                    
                    index = torch.all(batches == batch, dim=-1)
                    index_X = batches_offset_X[index]
                    index_X = (index_X[:,None] + torch.arange(batch[0])).flatten()

                    index_y = batches_offset_y[index]
                    index_y = (index_y[:,None] + torch.arange(batch[1])).flatten()
                
                    X_batch = X[index_X,:].reshape(sum(index), batch[0], 8)
                    y_batch = y[index_y,:].reshape(sum(index), batch[1], -1)
                    n_batch = batches[index]
                    
                    X_batch = self.transform_coordinate_batch(X_batch)
                
                    data_augmented.extend(list(zip(X_batch,y_batch,n_batch)))
                
                return data_augmented
            
            else:
                raise AssertionError("Invalid index!")
        
        else:
            return self.data[index]


def collect_fn(batch):
    X,y,n = zip(*batch)
    X = torch.cat(X)
    y = torch.cat(y)
    n = torch.stack(n)
    return (X,y,n)
    
    
class GNN_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super().__init__(dataset = dataset, 
                         batch_size = batch_size, 
                         shuffle = shuffle,
                         drop_last = drop_last,
                         collate_fn = collect_fn,
                         num_workers = 0,
                         )
