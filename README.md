#  Multi-Agent-Control
## Multi Agent Navigation in Unconstrained Environments using a Centralized Attention based Graphical Neural Network Controller

This repository contains code for the paper **Multi Agent Navigation in Unconstrained Environments using a Centralized Attention based Graphical Neural Network Controller** 

In this work, we propose a learning based neural model that provides control commands to simultaneously navigate multiple vehicles. The goal is to ensure that each vehicle reaches a desired target state without colliding with any other vehicle or obstacle in an unconstrained environment. The model utilizes an attention based Graphical Neural Network paradigm that takes into consideration the state of all the surrounding vehicles to make an informed decision. This allows each vehicle to smoothly reach its destination while also evading collision with the other agents. The data and corresponding labels for training such a network is obtained using an optimization based procedure. Experimental results demonstrates that our model is powerful enough to generalize even to situations with more vehicles than in the training data. Our method also outperforms comparable graphical neural network architectures.


## Environment

Clone the repo and build the conda environment:

```
conda create -n <env_name> python=3.7 
conda activate <env_name>
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install scipy
pip install --no-index torch-sparse --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install --no-index torch-cluster --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install --no-index torch-spline-conv --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-geometric==2.0.4
pip install pyyaml
pip install matplotlib
```


## Results of our Model 

Here we show results of our model in 4 different scenarios.

<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
          Scenario 1    
        </text>
    </td>
    <td width="50%">
       <text>
          Scenario 2    
        </text>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./images/IterGNN_MyTransformerConv_1.gif">
    </td>
    <td>
      <img src="./images/IterGNN_MyTransformerConv_2.gif">
    </td>
  </tr>
  <tr>
    <td width="50%">
        <text>
          Scenario 3    
        </text>
    </td>
    <td width="50%">
       <text>
          Scenario 4    
        </text>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./images/IterGNN_MyTransformerConv_3.gif">
    </td>
    <td>
      <img src="./images/IterGNN_MyTransformerConv_4.gif">
    </td>
  </tr>
</table>


### Attention Mechanism of our Model

The GIF below shows how the attention changes when the vehicles are moving. We visualize the mean of attention logits from all the graph attention layers of our model. The rows in the attention matrix correspond to the vehicle of interest. The columns show which vehicles/obstacle is being attended to. A lighter shade in the attention matrix depicts high attention and a darker shade represents lack of attention.

![image](./images/IterGNN_MyTransformerConv_Show_Attention.gif)


### Comparison with Other Models

Here we show the comparison of our result with  **GAINet**[1] **TransformerConv**[2], and **EdgeConv**[3] for two different scenarios.

As can be seen, only our model is capable of simultaneously driving all the vehicles to their desired destinations without collision. For all other models, the vehicles collide with each other. 


Example 1:

<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
          <strong>Our Model</strong>      
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_MyTransformerConv_3.gif">
        </figure>
    </td>
  </tr>
</table>
<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
        GAINet         
        </text> 
    </td>
    <td width="50%">
        <text>
        TransformerConv
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_GAINet_3.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_TransformerConv_3.gif">
        </figure>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <text>
        EdgeConv        
        </text>
    </td>
    <td width="50%">
        <text>
        No Graph        
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_EdgeConv_3.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_MyTransformerConv_NoEdge_3.gif">
        </figure>
    </td>
  </tr>
</table>


Example 2:

<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
          <strong>Our Model</strong>      
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_MyTransformerConv_4.gif">
        </figure>
    </td>
  </tr>
</table>
<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
        GAINet         
        </text> 
    </td>
    <td width="50%">
        <text>
        TransformerConv
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_GAINet_4.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_TransformerConv_4.gif">
        </figure>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <text>
        EdgeConv        
        </text>
    </td>
    <td width="50%">
        <text>
        No Graph        
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_EdgeConv_4.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/IterGNN_MyTransformerConv_NoEdge_4.gif">
        </figure>
    </td>
  </tr>
</table>


[1]: Liu, Y., Qi, X., Sisbot, E. A., and Oguchi, K. **Multi-agent trajectory prediction with graph attention isomorphismneural network**. In 2022 IEEE Intelligent Vehicles Symposium (IV), pp. 273–279, 2022. doi: 10.1109/IV51971.2022.9827155.

[2]: Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., and Sun,Y. **Masked label prediction:  Unified message passing model for semi-supervised classification**. arXiv preprint arXiv:2009.03509, 2020.

[3]: Yue, W., Yongbin, S., Ziwei, L., Sarma, S. E., and Bronstein, M. M. **Dynamic graph cnn for learning on point clouds**. Acm Transactions On Graphics (tog), 38(5), 2019.


## Tool for Visualizing Attention 

Run visualize_attention.py:
```
conda activate <env_name>
cd <path_to_this_repo>
python visualize_attention.py
```

A window will pump out:
![image](./images/Attention_Visualization_Tool.png)


To change the position of the vehicle/destination/obstacle, left click on it and move the mouse. 

To change the orientation of the vehicle/destination, left click on it and move the scroll wheel of the mouse.

To change the velocity of a vehicle, move the slider of the bar corresponding to that vehicle.

To change the size of the obstacle, left click on it and move the scroll wheel of the mouse.


You can also change the number of vehicles and obstacles in the scene by modifying the list [here](./configs/visualize_attention.yaml#L23). By default, it is 5 vehicles and 0 obstacle.


## Supplementary 

For the interested reader, we have provided addtional details in the supplementary material [Here](./supplementary.pdf). The supplementary contains the following 

- **Details of the U-Net inspired attention mechanism** 

- **Ablation studies on the contribution of the different components of the model** 

- **Run time comparison in our supplementary**

- **Breakdown of the training data**

- **Conservative optimization behaviour**