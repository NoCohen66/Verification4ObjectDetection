from detection.NeuralNetwork_LARD import torch_model_seq, torch_model_brightness, CustomModel, Neural_network_LARD, Neural_network_LARD_BrightnessContrast
from detection.perturbation import bound_whitenoise, bound_brightness_LARD, bound_contrast
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
torch_model = torch_model_seq
torch_model.load_state_dict(model_torch_load.state_dict())
#torch.Size([1, 3, 256, 256])
random_tensor = torch.randn(1,3,256,256)
print("Trying torch_model_seq", torch_model(random_tensor))
model_LARD= Neural_network_LARD()
print("Trying Neural_network_LARD", model_LARD(random_tensor))


model_bri = CustomModel(torch_model_seq)
model_bri.linear.weight.data = torch.zeros((3*256*256, 1), dtype=torch.float32)
model_bri.linear.bias.data = torch.zeros(3*256*256, dtype=torch.float32)
grr= torch.Tensor([0])
print(grr.shape)
output = model_bri(grr)
print("model_bri",model_bri(grr))

for name, param in model_bri.named_parameters():
     print(name, param.shape)

'''
model_corners = torch_model_brightness
corner_config = model_torch_load.state_dict()
corner_config['linear_perturbation.weight']=torch.Tensor(np.zeros((256*256, 1), dtype='float32'))
corner_config['linear_perturbation.bias']=torch.Tensor(np.zeros((256*256,), dtype='float32'))
model_corners.load_state_dict(corner_config)
'''
"""
new_layer = nn.Linear(1,256*256)
model_bri = nn.Sequential(new_layer, *torch_model)
corner_config_bri = model_bri.state_dict()
corner_config_bri['0.weight']=torch.Tensor(np.zeros((3*256*256, 1), dtype='float32'))
corner_config_bri['0.bias']=torch.Tensor(np.zeros((3*256*256,), dtype='float32'))
model_bri.load_state_dict(corner_config_bri)
"""
#model_bri(torch.tensor([[0.0]]))

"""
original_state_dict = model_torch_load.state_dict()
new_state_dict = {}
new_state_dict['0.weight'] = torch.zeros(256*256, 1)
new_state_dict['0.bias'] = torch.zeros(256*256)
for key in original_state_dict.keys():
    layer_num = int(key.split('.')[0]) + 1
    new_key = '{}.{}'.format(layer_num, key.split('.')[1])
    new_state_dict[new_key] = original_state_dict[key]
model_corners.load_state_dict(new_state_dict, strict=False)
model_corners(torch.tensor([[1.0]]))
"""
"""
model_corners_contrast = Neural_network_LARD_BrightnessContrast()
corner_config_contrast = model_torch_load.state_dict()
corner_config_contrast['linear_perturbation.weight']=torch.Tensor(np.zeros((256*256, 1), dtype='float32'))
corner_config_contrast['linear_perturbation.bias']=torch.Tensor(np.zeros((256*256,), dtype='float32'))
model_corners_contrast.load_state_dict(corner_config_contrast)
"""
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
eps_list = np.linspace(0, 0.0002,10)
eps_list_contrast = np.linspace(0, 0.01,101)


#for image_id in range(len(X_train)):
for image_id in range(1):
    print("image", image_id)
    list_info = []
    print(f"Begin to work with image {image_id}")
    st_im = time.time()

    X = X_train_[-image_id][None]/255
    print("hhhhhere", X.shape)
    y = Y_train_[-image_id]*256
    gt_box = y
    gt_box = gt_box.detach().numpy()
    ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])
 
    for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
        print("method", method)
        for i in range(len(eps_list)):
            print("variations", eps_list[i])
            start_perturbation = time.time() 
            lb_box_wn, ub_box_wn = bound_whitenoise(torch_model, X, eps_list[i], method=method.split()[0])
            lb_box_bri, ub_box_bri = bound_brightness_LARD(model_bri, X, eps_list[i], method=method.split()[0])
            #lb_box_contr, ub_box_contr = bound_contrast(model_corners_contrast, X, eps_list_contrast[i], method=method.split()[0])
            end_perturbation = time.time()
            st_computed_ious = time.time()
            perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn]}
            #"brightness":[lb_box_bri, ub_box_bri]} 
            #"contrast":[lb_box_contr, ub_box_contr]}
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
                                    "eps":eps_list[i], 
                                    "eps_contrast":eps_list_contrast[i],
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
    df.to_csv(f"results/LARD/new_{image_id}_iou_calculations.csv")
  

pd.DataFrame(list_info_time).to_csv("results/LARD/times2.csv")
pd.DataFrame(images_exp).to_csv("results/LARD/images.csv")

