from detection.NeuralNetwork_OL_v2 import NeuralNetwork_OL_v2
from detection.CustomMnistDataset_OL import CustomMnistDataset_OL
from detection.perturbation import bound_whitenoise
from iou_calculator.Hyperrectangle_interval import Hyperrectangle_interval
from iou_calculator.Hyperrectangle import Hyperrectangle
from iou_calculator.Interval import Interval



import torch 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time 


filename = 'detection/weights/toy_model_classif'
model_torch_load  = torch.jit.load(f'{filename}.pt')
model_digit = NeuralNetwork_OL_v2(classif=True)
model_digit.load_state_dict(model_torch_load.state_dict())

filename = 'detection/weights/toy_model_corners'
model_torch_load  = torch.jit.load(f'{filename}.pt')
model_box = NeuralNetwork_OL_v2(classif=False)
model_box.load_state_dict(model_torch_load.state_dict())

train_df = pd.read_csv("detection/MNIST/train.csv")
trainingData = CustomMnistDataset_OL(train_df)
train_dataloader = DataLoader(trainingData, batch_size=1, shuffle=True)
train_iterator = iter(train_dataloader)

list_info = []
list_info_time = []
eps_list = [eps_i/255/4. for eps_i in range(1,40)]


for image_id in range(4):

    print(f"Begin to work with image {image_id}")
    
    data = next(train_iterator)
    X, y = data
    gt_logit, gt_box = y
    gt_box = gt_box.detach().numpy()[0]
    print(gt_box)
    for eps in  eps_list:
        st = time.time()
        lb_box, ub_box, lb_adv, ub_adv = bound_whitenoise(model_box, model_digit, X, eps)

        print("lb_box:", lb_box)
        Hyperrectangle_interval(x_1=Interval(lb_box[0],ub_box[0]), x_2=Interval(lb_box[2],ub_box[2]), y_1=Interval(lb_box[1],ub_box[1]), y_2=Interval(lb_box[3],ub_box[3]))
        et = time.time()
        list_info.append({"image_id":image_id, 
                            "eps":eps, 
                            "perturbation":"whitenoise", 
                            "elapsed_time_eps" : et - st })
    
    

#pd.DataFrame(list_info).to_csv("bound_results/results.csv")

