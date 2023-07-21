import os
import torch
import numpy as np

from data_process import get_problem, load_yaml

from argparse import ArgumentParser


def generate_test_data(config):
        
    
    problem_collection = np.array(config["problem collection"], dtype=int)
    assert problem_collection.shape[1] == 2 and \
        len(problem_collection.shape) == 2 and \
        np.amin(problem_collection[:,0]) >= 1 and \
        np.amin(problem_collection[:,1]) >= 0, \
        "Invalid input of problem_collection!"
    
    os.makedirs(config["data folder"], exist_ok=True)
    
    for len_vehicle, len_obstacle in problem_collection:
        
        print(f"Generating Test Dataset {len_vehicle} Vehicle {len_obstacle} Obstalce.")
        
        vehicles, obstacles = get_problem(len_vehicle, 
                                            len_obstacle, 
                                            data_length = config["data length each case"],
                                            collision = config["collision mode"],
                                            parking = config["parking mode"],
                                            mode = "generate test problem",
                                            )

        test_data = torch.from_numpy(np.concatenate((vehicles, obstacles), axis=1))
        test_data_path = os.path.join(config["data folder"], 
                                        f"test_data_vehicle={len_vehicle}_obstalce={len_obstacle}.pt")
        torch.save(test_data, test_data_path)


if __name__ == "__main__":
    
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/generate_test_data.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    generate_test_data(config)
