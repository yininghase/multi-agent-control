import os
import time
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from simulation import sim_run
from gnn import IterativeGNNModel
from data_process import get_problem, load_test_data, load_yaml


def inference(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available now:', device)
    
    problem_collection = np.array(config['problem collection'])
    assert problem_collection.shape == (1,2) and \
        np.amin(problem_collection[:,0]) >= 1 and \
        np.amin(problem_collection[:,1]) >= 0, \
        "Invalid input of problem_collection!"

    if config["collect data"]:
        data = {}
        
        if config["collect trajectory"]:
            config["random offset"] = False
        
        if not os.path.exists(config["data folder"]):
            os.makedirs(config["data folder"])
        
        for problem in problem_collection:
            
            num_vehicles, num_obstacles = problem
        
            data[(num_vehicles, num_obstacles)] = {
                        "X_data": torch.tensor([]), 
                        "y_model_data": torch.tensor([]), "batches_data": torch.tensor([]),
                        "X_data_path": os.path.join(config["data folder"], 
                                                    f"X_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                        "y_model_data_path": os.path.join(config["data folder"], 
                                                        f"y_model_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                        "batches_data_path": os.path.join(config["data folder"], 
                                                        f"batches_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"), 
                    }
            
            if config["collect trajectory"]:
                data[(num_vehicles, num_obstacles)].update({"trajectory_data": torch.tensor([0]),
                                                            "trajectory_data_path": os.path.join(config["data folder"], 
                                                            f"trajectory_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                            })

    assert config["horizon"] == 1, "keep horizon as 1 for GNN inference!"

    model = IterativeGNNModel(horizon = config["horizon"],  
                            max_num_vehicles = problem_collection[0,0], 
                            max_num_obstacles = problem_collection[0,1],
                            mode = "inference",
                            device = device,
                            conv_type = config["convolution type"],
                            )
    
    model.load_state_dict(torch.load(config["model path"]))
    model.to(device)
    
    
    if config["test data souce"] == "fixed test data":
        
        assert os.path.exists(config["test data folder"]), \
            "The test data folder does not exist!"
        
        test_data = load_test_data(num_vehicles = problem_collection[0,0],
                                    num_obstacles = problem_collection[0,1],
                                    load_all_simpler = False, 
                                    folders = config["test data folder"],
                                    lim_length = config["test data each case"],
                                    )

        config["simulation runs"] = len(test_data)
    
    elif config["test data souce"] == "on the fly":
        pass
    
    else:
        raise NotImplementedError("Unknown test data source!")
    
    time_list = []
    num_step_list = []

    for i in tqdm(range(config["simulation runs"])):
        
        if config["test data souce"] == "fixed test data":
            
            starts, targets, obstacles, (num_vehicles, num_obstacles) = test_data[i]
            
            starts = starts.numpy()
            targets = targets.numpy()
            obstacles = obstacles.numpy()
            num_vehicles = num_vehicles.item()
            num_obstacles = num_obstacles.item()
            
        
        elif config["test data souce"] == "on the fly":
        
            num_vehicles, num_obstacles = problem_collection[int(i%len(problem_collection))]            
            starts, targets, obstacles = get_problem(num_vehicles, num_obstacles, 
                                                collision = config["collision mode"], 
                                                parking = config["parking mode"],
                                                mode = "inference")
            
        model_name = os.path.basename(config["model path"]).split(".")[0]
        
        if config["convolution type"].split('_')[-1] == "NoVehicleEdges":
            model_name += "_NoVehicleEdges"
            
        config["start"] = starts
        config["target"] = targets
        config["obstacles"] = obstacles
        config["name"] = f"{model_name}_vehicle={num_vehicles}_obstacle={num_obstacles}_run={i}"
        config["num of vehicles"] = num_vehicles
        config["num of obstacles"] = num_obstacles
        
        start_time = time.time()
        X_data, batches_data, _, y_model_data, success, num_step = sim_run(config, model = model, device = device)
        end_time = time.time()
        
        time_list.append(end_time-start_time)
        num_step_list.append(num_step)
        
        if config["collect data"]:
            
            data[(num_vehicles, num_obstacles)]["X_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["X_data"], X_data))
            data[(num_vehicles, num_obstacles)]["y_model_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["y_model_data"], y_model_data))
            data[(num_vehicles, num_obstacles)]["batches_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["batches_data"], batches_data))
            if config["collect trajectory"]:
                data[(num_vehicles, num_obstacles)]["trajectory_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["trajectory_data"], 
                                                                                    torch.tensor([data[(num_vehicles, num_obstacles)]["trajectory_data"][-1]+len(X_data)])))
            
            torch.save(data[(num_vehicles, num_obstacles)]["X_data"], data[(num_vehicles, num_obstacles)]["X_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["y_model_data"], data[(num_vehicles, num_obstacles)]["y_model_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["batches_data"], data[(num_vehicles, num_obstacles)]["batches_data_path"])
            if config["collect trajectory"]:
                torch.save(data[(num_vehicles, num_obstacles)]["trajectory_data"], data[(num_vehicles, num_obstacles)]["trajectory_data_path"])

    time_list = np.array(time_list)
    num_step_list = np.array(num_step_list)
    
    average_run_time_per_task = np.mean(time_list)
    print(f"Average run time per task: {average_run_time_per_task}")
    average_run_time_per_step = np.sum(time_list)/np.sum(num_step_list)
    print(f"Average run time per step: {average_run_time_per_step}")
        
        

if __name__ == "__main__":     
    
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/inference.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    problem_collection = config['problem collection'].copy()
    model_name = os.path.basename(config["model path"]).split(".")[0]
    
    if config["convolution type"].split('_')[-1] == "NoVehicleEdges":
        model_name += "_NoVehicleEdges"
    
    config["plot folder"] = os.path.join(config["plot folder"], model_name)
    config["data folder"] = os.path.join(config["data folder"], model_name)
    
    for i in range(len(problem_collection)):
        print(f"current task: num_vehicle={problem_collection[i][0]}, num_obstacle={problem_collection[i][1]}")
        config['problem collection'] = [problem_collection[i]]
        inference(config)