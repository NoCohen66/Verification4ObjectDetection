from detection.NeuralNetwork_OL_v2 import NeuralNetwork_OL_v2
from detection.CustomMnistDataset_OL import CustomMnistDataset_OL
from detection.perturbation import bound_whitenoise
from iou_calculator.Hyperrectangle_interval import Hyperrectangle_interval
from iou_calculator.Hyperrectangle import Hyperrectangle
from iou_calculator.Interval import Interval
from iou_calculator.IoU import IoU
from iou_calculator.utils import Merge, show_im, show_im_origin, check_box, clip_corner
from IPython.display import display


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
eps_list = [eps_i/255 for eps_i in range(0,40)]


for image_id in range(1000):
    print(f"Begin to work with image {image_id}")
    st_im = time.time()
    data = next(train_iterator)
    X, y = data
    gt_logit, gt_box = y
    gt_box = gt_box.detach().numpy()[0]
    #check_gt_box(gt_box)
    #show_im_origin(X, f"images/{image_id}_gt.png", gt_box)

    for i in range(len(eps_list)):
        st_computed_ious = time.time()
        lb_box, ub_box, lb_adv, ub_adv = bound_whitenoise(model_box, model_digit, X, eps_list[i])
        for corner in lb_box: 
            corner = clip_corner(corner)
        for corner in ub_box:
            corner = clip_corner(corner)
        end_perturbation = time.time()
        if check_box(lb_box) and check_box(ub_box):
            try:
                ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])
                predicted_box = Hyperrectangle_interval(x_1=Interval(lb_box[0],ub_box[0]), x_2=Interval(lb_box[2],ub_box[2]), y_1=Interval(lb_box[1],ub_box[1]), y_2=Interval(lb_box[3],ub_box[3]))
                dict_iou = IoU(predicted_box, ground_truth_box).iou(display = False)
                et_computed_ious = time.time()
                list_info.append(Merge({"image_id":image_id, 
                                    "gt_logit": gt_logit,
                                    "eps":eps_list[i], 
                                    "perturbation":"whitenoise", 
                                    "elapsed_time_perturbation":st_computed_ious-end_perturbation,
                                    "elapsed_time_eps_computed_ious" : et_computed_ious - st_computed_ious }, dict_iou))
            except ValueError as e:
                print(e)
            
        else:
            pass
        

 
    et_im = time.time()
    list_info_time.append((image_id, et_im-st_im))
    df = pd.DataFrame(list_info)
    df.to_csv(f"results/{gt_logit.item()}/{image_id}_{df.shape[0]}_{et_im-st_im}seconds_.csv")
  

#pd.DataFrame(list_info).to_csv("bound_results/results.csv")

