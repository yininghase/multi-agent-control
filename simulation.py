import random
import numpy as np
import torch
from scipy.optimize import minimize

from mpc import ModelPredictiveControl
from visualization import Visualize_Trajectory
from data_process import change_to_relative_frame, get_angle_diff

def sim_run(simulation_options, model = None, device = 'cpu'):
    
    mpc = ModelPredictiveControl(simulation_options)

    if mpc.control_init is None:
        u = np.zeros((mpc.horizon, mpc.num_vehicle, 2))
    else:
        u = mpc.control_init[:mpc.horizon]
    
    # we have limit on steering angle and pedal   
    bounds = np.array([[-1, 1], [-0.8, 0.8]])
    bounds = np.tile(bounds, (mpc.horizon*mpc.num_vehicle, 1))

    ref = mpc.target

    state_i = np.array([mpc.start])

    sim_total = simulation_options["simulation time"]
    
    predict_info_opt = np.empty((0, mpc.horizon+1, mpc.num_vehicle, 4)) # to store predicted states from MPC
    predict_info_model = np.empty((0, mpc.horizon+1, mpc.num_vehicle, 4)) # to store predicted states from Model

    label_data_opt =  np.empty((0, mpc.horizon, mpc.num_vehicle, 2)) # to store predicted controls from MPC
    label_data_model = np.empty((0, mpc.horizon, mpc.num_vehicle, 2)) # to store predicted controls from Model
    
    # attention = [np.zeros((mpc.num_vehicle, mpc.num_vehicle+mpc.num_obstacle))]

    # range for random offset 
    offsets = [1.0, 1.001]
    offset = offsets[random.randint(0, 1)]

    for i in range(1, sim_total+1):
        ### Optimization Prediction from MPC ###
        if simulation_options["show optimization"]:
            u = u[1:,...]
            # u = np.concatenate((u, np.zeros((1, mpc.num_vehicle, 2))), axis=0) 
            u = np.concatenate((u, u[-1][None,...]), axis=0)      
            u_solution = minimize(mpc.cost_function, u.flatten(), (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol=1e-5)
            u_opt = u_solution.x.reshape(mpc.horizon, mpc.num_vehicle, 2)
            y_opt = mpc.plant_model(state_i[-1], mpc.dt, u_opt[0])
            label_data_opt = np.concatenate((label_data_opt, u_opt[None,...]))
            predict_info_opt = np.concatenate((predict_info_opt, get_predictions(mpc, state_i[-1], u_opt)[None,...]))
            u = u_opt

        ### Model Prediction ###
        if model is not None:
            model.eval()
            
            vehicles = torch.tensor(np.concatenate((state_i[-1], np.array(ref)), axis=1), dtype=torch.float32)
            vehicles = torch.cat([vehicles, torch.zeros(mpc.num_vehicle, 1)], dim=1)
            
            if mpc.num_obstacle > 0:
                obstacles = torch.tensor(simulation_options["obstacles"],dtype=torch.float32)
                obstacles = torch.cat((torch.zeros(mpc.num_obstacle,4), obstacles, torch.ones(mpc.num_obstacle, 1)), dim=1)
                model_input = torch.cat((vehicles, obstacles), dim=0)
            else:
                model_input = vehicles
            
            obstacles = slice(mpc.num_vehicle, mpc.num_vehicle+mpc.num_obstacle)
            model_input[obstacles,:2] = model_input[obstacles,4:6]
            model_input[obstacles,-1] = model_input[obstacles,-2]
            model_input[obstacles,-2] = 0
            
            model_input = model_input.to(device)
            batches = torch.tensor([[mpc.num_vehicle+mpc.num_obstacle, mpc.num_vehicle]], dtype=torch.int, device=device)
            u_model = model(model_input, batches)[0].detach().cpu().numpy()
            # attention.append(np.mean(model.attention.detach().cpu().numpy(), axis=0))
                
            u_model = u_model.reshape(mpc.num_vehicle, mpc.horizon, 2)
            u_model = np.transpose(u_model, (1,0,2))
            
            y_model = mpc.plant_model(state_i[-1], mpc.dt, u_model[0])
            label_data_model = np.concatenate((label_data_model, u_model[None,...]))
            predict_info_model = np.concatenate((predict_info_model, get_predictions(mpc, state_i[-1], u_model)[None,...]))
        
        ### Simulation ###
        y = None           
        if model is not None:
            y = y_model[None,...]
        else:
            y = y_opt[None,...]

        if simulation_options["random offset"]:
            state_quotient = (i/sim_total) if i >= 10 else 1
            y = introduce_random_offset(y, mpc.num_vehicle, state_quotient, base_offset=offset)

        state_i = np.concatenate((state_i, y), axis=0)
        
        # check if the vehicle has not moved the last n steps and if this is not the case end the simulation
        n = 10 # if the vehicle has not moved more then "stop tolerance" for n steps we end the simulation
        if len(state_i) > n and \
           np.all(np.sum(np.linalg.norm(np.diff(state_i[-n:,:,:2], axis=0), axis=-1), axis=0) < simulation_options["stop tolerance"]) and \
           np.all(np.sum(get_angle_diff(np.diff(state_i[-n:,:,2], axis=0), 0), axis=0) < 2*simulation_options["stop tolerance"]):
            break
    
    num_step = len(state_i)-1
    
    # check if all the vehicle reach their goals at the end of the simulation with in tolerance,
    # if not, the simulation failed, the data should not be collected
    if  model is None and simulation_options["collect data"] and \
        not (np.all(np.linalg.norm(state_i[-1:,:,:2] - ref[:,:2], axis=-1) < simulation_options["position tolerance"]) and \
            np.all(get_angle_diff(state_i[-1:,:, 2], ref[:,2]) < simulation_options["angle tolerance"])) :
        
        print(f"max position error: {np.max(np.linalg.norm(state_i[-1:,:,:2] - ref[:,:2], axis=-1)):.6f}")
        print(f"max angle error: {np.max(np.abs(state_i[-1:,:, 2] - ref[:,2])):.6f}")
        
        # return  None, None, None, None, False
        success = False
    
    else:
        success = True
    
    
    if simulation_options["save plot"] or simulation_options["show plot"]:
        simulation_options["is model"] = (model is not None)
        # visualization = Visualization(simulation_options, show_attention=True)
        # visualization.create_video(state_i, predict_info_opt, predict_info_model, attention)
        visualization = Visualize_Trajectory(simulation_options)
        visualization.create_video(state_i, predict_info_opt, predict_info_model)
        visualization.plot_trajectory(state_i)
        
    ###################
    # COLLECTING TRAINING DATA
    if simulation_options["collect data"]:

        # For each data point of vehicle we have a vehicle position (x,y), angle and velocity and the desired position and angle, 
        # For each data point of obstacle we have a obstacle position (x,y), radius 
        # To distinguish the vehicle and obstacle, we add a sign addtionally, for vehicle it is 0, for obstacle it is 1
        # so for vehicle: [x, y, angle, v, x_d, y_d, angle_d, 0], for obstacle: [0, 0, 0, 0, x, y, r, 1]
        # for the problem of m vehicles and n obstacles, we stack the first m vehicles and n obstacles together like
        # [[x, y, angle, v, x_d, y_d, angle_d, 0], # vehicle 1 
        #             ......
        #  [x, y, angle, v, x_d, y_d, angle_d, 0], # vehicle m
        #  [0, 0, 0, 0, x, y, r, 1],   # obstacle 1
        #             ......
        #  [0, 0, 0, 0, x, y, r, 1],   # obstacle n
        # ] 
        # 
        vehicles = np.concatenate((state_i[:-1], np.tile(ref, (len(state_i)-1, 1, 1)), np.zeros((len(state_i)-1,mpc.num_vehicle,1))), axis=-1)
        if mpc.num_obstacle > 0:
            obstacles = simulation_options["obstacles"]
            obstacles = np.concatenate((np.zeros((mpc.num_obstacle, 4)), obstacles, np.ones((mpc.num_obstacle,1))), axis=-1)
            X_tensor = np.concatenate((vehicles, np.tile(obstacles, (len(state_i)-1, 1, 1))), axis=1)
        else:
            X_tensor = vehicles
        
        X_tensor = torch.tensor(X_tensor.reshape(-1,8))
        
        # Each ground truth label of vehicle data points has to contain the predicted steps of 
        # pedal and steering angle at each step of the horizon, like
        # [[pedal_0, steering_angle_0, pedal_1, steering_angle_1, ..., pedal_t, steering_angle_t]  # vehicle 1 
        #  ...
        #  [pedal_0, steering_angle_0, pedal_1, steering_angle_1, ..., pedal_t, steering_angle_t]  # vehicle m
        # ]
        # for obstacle data points there is no ground truth label
        y_tensor_model = torch.tensor(label_data_model)
        if len(y_tensor_model) > 0:
            y_tensor_model = torch.transpose(y_tensor_model, 1, 2).reshape(-1, mpc.horizon*2)
        
        y_tensor_GT = torch.tensor(label_data_opt)
        y_tensor_GT = torch.transpose(y_tensor_GT, 1, 2).reshape(-1, mpc.horizon*2)
        
        # we need additional tensor to record the size of the problem, 
        # i.e, how many vehicles and obstacles are included in the case
        # for a simulation time of T, we can get datas of T problems
        # so the batch tensor is like
        # [[num_vehicles + num_obstacles, num_vehicles], # problem 1
        #           ......
        #  [num_vehicles + num_obstacles, num_vehicles], # problem T
        # ]
        batches_tensor = torch.tensor([[mpc.num_vehicle+mpc.num_obstacle, mpc.num_vehicle]]).repeat(len(state_i)-1, 1)
        
    else:
        X_tensor = None
        y_tensor_GT = None
        y_tensor_model = None
        batches_tensor = None
    
    return X_tensor, batches_tensor, y_tensor_GT, y_tensor_model, success, num_step # simulation_sucessfull

def get_predictions(mpc, initial_state, u):
    
    predicted_state = np.array([initial_state])
    for i in range(mpc.horizon):
        predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[i])
        predicted_state = np.concatenate((predicted_state, predicted[None,...]))
    
    return predicted_state

def introduce_random_offset(y, num_vehicle, state_quotient, base_offset=1):

    sigma = np.array([0.25,0.25,np.pi/18,0.25])
    offset = np.random.normal(0, sigma*(base_offset-state_quotient), (num_vehicle,4))
    
    return y + offset

