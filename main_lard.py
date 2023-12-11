from detection.NeuralNetwork_LARD import Neural_network_LARD, Neural_network_LARD_BrightnessContrast
from detection.perturbation import bound_whitenoise, bound_brightness_LARD, bound_contrast_LARD
from iou_calculator.Hyperrectangle_interval import Hyperrectangle_interval
from iou_calculator.Hyperrectangle import Hyperrectangle
from iou_calculator.Interval import Interval
from iou_calculator.IoU import IoU
from iou_calculator.utils import Merge, show_im, show_im_origin, check_box, clip_corner
from IPython.display import display
from contextlib import closing
import pickle as pkl

import torch 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time 
from torch import nn
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
filename = 'detection/weights/tmp_nfm_v4'
model_torch_load  = torch.load(f'{filename}.pt', torch.device('cpu')) # change if GPU
torch_model = Neural_network_LARD()
dict_val = {}
for name, param in zip(torch_model.state_dict().keys(),
                model_torch_load.state_dict().values()):
     dict_val[name] = param

torch_model.load_state_dict(dict_val)
model_bri = Neural_network_LARD_BrightnessContrast()
bri_config = dict_val
bri_config['linear_perturbation.weight']=torch.Tensor(np.zeros((3*256*256, 1), dtype='float32'))
bri_config['linear_perturbation.bias']=torch.Tensor(np.zeros((3*256*256,), dtype='float32'))
model_bri.load_state_dict(bri_config)

model_contr = Neural_network_LARD_BrightnessContrast()
contr_config = dict_val
contr_config['linear_perturbation.weight']=torch.Tensor(np.zeros((3*256*256, 1), dtype='float32'))
contr_config['linear_perturbation.bias']=torch.Tensor(np.zeros((3*256*256,), dtype='float32'))
model_contr.load_state_dict(contr_config)


with closing((open('detection/LARD/lard_nfm_data_iou.pkl', 'rb'))) as f:
             dico_dataset = pkl.load(f)

X_train = dico_dataset['x_train']
X_test = dico_dataset['x_test']
Y_train = dico_dataset['y_train']
Y_test = dico_dataset['y_test']
X_train_ = torch.Tensor(X_train).to(device)
X_test_ = torch.Tensor(X_test).to(device)
Y_train_ = torch.Tensor(Y_train).to(device)
Y_test_ = torch.Tensor(Y_test).to(device)

print("Let's apply it to test dataset:", len(X_train))

list_info_time = []
images_exp = []
eps_list = np.linspace(0, 0.0002,11)
eps_list_contrast_CROWN = np.linspace(0, 0.1,11)
eps_list_contrast = np.linspace(0, 0.001,11)


for image_id in range(105):
    image_id = image_id + 1 #not dealing with image 0
    print("image", image_id)
    list_info = []
    print(f"Begin to work with image {image_id}")
    st_im = time.time()

    X = X_train_[-image_id][None]/255
    y = Y_train_[-image_id]*256
    gt_box = y
    gt_box = gt_box.detach().numpy()
    ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])
 
    for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)' ]:
        print("method", method)
        for i in range(len(eps_list)):
            print("variations", eps_list[i])
            start_perturbation = time.time() 
            if method == 'backward (CROWN)':
                 lb_box_contr, ub_box_contr = bound_contrast_LARD(model_contr, X, eps_list_contrast_CROWN[i], method=method.split()[0])
            else: 
                 lb_box_contr, ub_box_contr = bound_contrast_LARD(model_contr, X, eps_list_contrast[i], method=method.split()[0])
            
            end_perturbation = time.time()
            st_computed_ious = time.time()
            perturbations_dict = {"contrast":[lb_box_contr, ub_box_contr]}
            for perturbation_name, bounds in perturbations_dict.items():
                print("perturbation_name", perturbation_name)
                lb_box, ub_box = bounds[0], bounds[1]
                lb_box = [clip_corner(corner, LARD=True) for corner in lb_box]
                ub_box = [clip_corner(corner, LARD=True) for corner in ub_box]
                try: 
                    predicted_box = Hyperrectangle_interval(x_1=Interval(lb_box[0],ub_box[0]), x_2=Interval(lb_box[2],ub_box[2]), y_1=Interval(lb_box[1],ub_box[1]), y_2=Interval(lb_box[3],ub_box[3]))
                    dict_iou = IoU(predicted_box, ground_truth_box).iou(display = False)
                    fake_iou = False
                except ValueError as e: 
                    print(e)
                    dict_iou = {"IoU_vanilla":[0,1],
                            "tmps_vanilla": 0,
                            "IoU_extension":[0,1],
                            "tmps_extension":0}
                    fake_iou = True
                et_computed_ious = time.time()
            
                list_info.append(Merge({"method":method,
                                    "image_id":image_id, 
                                    #"eps":eps_list[i], 
                                    "eps_contrast":eps_list_contrast[i],
                                    "eps_contrast_CROWN":eps_list_contrast[i],
                                    "fake_iou": fake_iou,
                                    "perturbation":perturbation_name, 
                                    "bounds_clip":[lb_box, ub_box],
                                    "elapsed_time_perturbation":end_perturbation-start_perturbation,
                                    "elapsed_time_eps_computed_ious" : et_computed_ious - st_computed_ious }, dict_iou))

    et_im = time.time()
    print(f"Time to proceed one image {et_im-st_im}")
    list_info_time.append((image_id, et_im-st_im))
    images_exp.append(X[0,0,:,:].flatten().tolist())
    df = pd.DataFrame(list_info)
    df.to_csv(f"results/LARD/contr_{image_id}_iou_calculations.csv")
  

pd.DataFrame(list_info_time).to_csv("results/LARD/contr_times2.csv")
pd.DataFrame(images_exp).to_csv("results/LARD/contr_images.csv")

