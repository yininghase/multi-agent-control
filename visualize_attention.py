import os
import torch
import numpy as np

from argparse import ArgumentParser

from gnn import IterativeGNNModel
from data_process import load_yaml, get_problem, load_test_data
from visualization import Visualize_Attention


def inference(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device available now:', device)
    
    
    problem_collection = np.array(config['problem collection'])
    assert problem_collection.shape == (1,2) and \
        np.amin(problem_collection[:,0]) >= 1 and \
        np.amin(problem_collection[:,1]) >= 0, \
        "Invalid input of problem_collection!"
    
    model_name = os.path.basename(config["model path"]).split(".")[0]

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
        if len(test_data) > config["simulation runs"]:
            select_idx = np.random.choice(len(test_data), size=config["simulation runs"], replace=False)
            test_data = test_data[select_idx]   
        else:
            config["simulation runs"] = len(test_data)
    
    elif config["test data souce"] == "on the fly":
        pass
    
    else:
        raise NotImplementedError("Unknown test data source!")
    
    for i in range(config["simulation runs"]):

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
            
        starts[:,3] = np.random.uniform(low=-2, high=3, size=num_vehicles)
        
        model_name = os.path.basename(config["model path"]).split(".")[0]
        config["start"] = starts
        config["target"] = targets
        config["obstacles"] = obstacles
        config["name"] = f"{model_name}_vehicle={num_vehicles}_obstacle={num_obstacles}_run={i}"
        config["num of vehicles"] = num_vehicles
        config["num of obstacles"] = num_obstacles

        visualize_attention = Visualize_Attention(config, model, device)
        visualize_attention.base_plot(move_mode=True)
            

if __name__ == "__main__":    
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/visualize_attention.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    problem_collection = config['problem collection'].copy()
    
    for i in range(len(problem_collection)):
        print(f"current task: num_vehicle={problem_collection[i][0]}, num_obstacle={problem_collection[i][1]}")
        config['problem collection'] = [problem_collection[i]]
        inference(config)
    