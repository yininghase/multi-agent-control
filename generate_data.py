import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from simulation import sim_run
from data_process import get_problem, load_yaml

from argparse import ArgumentParser

warnings.filterwarnings("ignore")
    
def length(data):
    
    length = 0
    
    for value in data.values():
        batch = value["batches_data"]
        length += len(batch)
    
    return length
        

def generate_data(config):
    '''function to collect training data for supervised learning'''
    
    if not os.path.exists(config["data folder"]):
        os.makedirs(config["data folder"])
    
    # problem with different number of vehicles and obstacles
    # [[num_vehicles, num_obstacles], ...]
   
    problem_collection = np.array(config['problem collection'])
    assert problem_collection.shape[1] == 2 and \
            len(problem_collection.shape) == 2 and \
            np.amin(problem_collection[:,0]) >= 1 and \
            np.amin(problem_collection[:,1]) >= 0, \
            "Invalid input of problem_collection!"
        
    
    data = {}
        
    for problem in problem_collection:
        
        num_vehicles, num_obstacles = problem
        
               
        data[(num_vehicles, num_obstacles)] = {
                "X_data": torch.tensor([]), 
                "y_GT_data": torch.tensor([]), 
                "batches_data": torch.tensor([]),
                "X_data_path": os.path.join(config["data folder"], 
                                            f"X_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                "y_GT_data_path": os.path.join(config["data folder"], 
                                            f"y_GT_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                "batches_data_path": os.path.join(config["data folder"], 
                                                f"batches_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"), 
            }
        
        if config["collect trajectory"]:
            
            data[(num_vehicles, num_obstacles)].update({
                    "trajectory_data": torch.tensor([0]),
                    "trajectory_data_path": os.path.join(config["data folder"], 
                                                    f"trajectory_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                })


    
    for i in tqdm(range(config["simutaion runs"])):
        
        # show first 5 trajectories to check the quality of ground truth data
        if i >= 5:
            config["save plot"] = False
            
        num_vehicles, num_obstacles = problem_collection[int(i%len(problem_collection))]
        
        
        start, target, obstacles = get_problem(num_vehicles, num_obstacles, 
                                                collision = config["collision mode"],
                                                parking = config["parking mode"],
                                                mode = "generate train trajectory")
        
        config["start"] = start
        config["target"] = target
        config["obstacles"] = obstacles
        config["name"] = f"generate_data_{i}"
        config["num of vehicles"] = num_vehicles
        config["num of obstacles"] = num_obstacles

        X_data, batches_data, y_GT_data, _, success, num_step = sim_run(config) 
            
        if (not success):
            print(f"The current problem of {num_vehicles} vehicle(s) and {num_obstacles} obstacle(s) failed!")
            if (not config["save failed samples"]):
                print("The failed data is not saved!")
                continue
        
        data[(num_vehicles, num_obstacles)]["X_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["X_data"], X_data))
        data[(num_vehicles, num_obstacles)]["y_GT_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["y_GT_data"], y_GT_data))
        data[(num_vehicles, num_obstacles)]["batches_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["batches_data"], batches_data))
        
        if config["collect trajectory"]:
            data[(num_vehicles, num_obstacles)]["trajectory_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["trajectory_data"], 
                                                                                torch.tensor([data[(num_vehicles, num_obstacles)]["trajectory_data"][-1]+len(X_data)])))
        
        # save the data and the labels
        print(f"Saving data at step {i+1}")
        print(f"Datapoints collected: {length(data)}")
        
        torch.save(data[(num_vehicles, num_obstacles)]["X_data"], data[(num_vehicles, num_obstacles)]["X_data_path"])
        torch.save(data[(num_vehicles, num_obstacles)]["y_GT_data"], data[(num_vehicles, num_obstacles)]["y_GT_data_path"])
        torch.save(data[(num_vehicles, num_obstacles)]["batches_data"], data[(num_vehicles, num_obstacles)]["batches_data_path"])
        if config["collect trajectory"]:
            torch.save(data[(num_vehicles, num_obstacles)]["trajectory_data"], data[(num_vehicles, num_obstacles)]["trajectory_data_path"])
        
    for problem in problem_collection:
        
        num_vehicles, num_obstacles = problem
        if len(data[(num_vehicles, num_obstacles)]["X_data"]) > 0:
            torch.save(data[(num_vehicles, num_obstacles)]["X_data"], data[(num_vehicles, num_obstacles)]["X_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["y_GT_data"], data[(num_vehicles, num_obstacles)]["y_GT_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["batches_data"], data[(num_vehicles, num_obstacles)]["batches_data_path"])
            if config["collect trajectory"]:
                torch.save(data[(num_vehicles, num_obstacles)]["trajectory_data"], data[(num_vehicles, num_obstacles)]["trajectory_data_path"])   


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/generate_data.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    generate_data(config)
   



    
    