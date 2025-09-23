# ------------------------------------------------------------------------------
# Written by Yiwen Bai (wen1109@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------

import _init_paths
import models
from thop import profile
from configs import config
from configs import update_config
from models.UCMNetVisualizer import *
import torch


# 1. Load model and set it to GPU
model = models.ucmnet.get_seg_model(config, imgnet_pretrained=True)
model.load_state_dict(torch.load('/root/autodl-tmp/UCMNet/output/railsem19/ucmnet_railsem19/best.pt'), strict=False)
# model.load_state_dict(torch.load('/root/autodl-tmp/UCMNet/output/railsem19/ucmnet_railsem19_1536x768/best.pt'), strict=False)
model = model.cuda()  # Move the model to GPU

# 2. Initialize the visualizer
visualizer = UCMNetVisualizer(model, device='cuda')

# 3. Preprocess the image
input_tensor, original_img = preprocess_image('/root/autodl-tmp/UCMNet/rs04973.png')

# Ensure the input tensor is on the same device as the model (GPU)
input_tensor = input_tensor.cuda()

# 4. Define target layers for visualization
target_layers = [
    'conv1',           # Initial convolution
    # 'layer1',          # I branch layer 1
    # 'layer2',          # I branch layer 2
    # 'layer3',          # I branch layer 3
    # 'layer4',          # I branch layer 4
    # 'layer5',          # I branch layer 5
    # 'layer3_',         # P branch layer 3
    # 'layer4_',         # P branch layer 4
    # 'layer5_',         # P branch layer 5
    # 'layer3_d',        # D branch layer 3
    # 'layer4_d',        # D branch layer 4
    # 'layer5_d',        # D branch layer 5
    'spp',        # D branch layer 5
]

# 5. Visualize activations
features = visualizer.visualize_layer_activations(
    input_tensor, 
    original_img, 
    target_layers, 
    save_dir='./ucmnet_visualizations'
)
