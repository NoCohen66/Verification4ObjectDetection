from detection.NeuralNetwork_OL_v2 import NeuralNetwork_OL_v2
from detection.CustomMnistDataset_OL import CustomMnistDataset_OL
from detection.NeuralNetwork_BrightnessContrast import NeuralNetwork_BrightnessContrast
from detection.perturbation import bound_whitenoise, bound_brightness
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

model_corners = NeuralNetwork_BrightnessContrast(classif=False)
corner_config = model_torch_load.state_dict()
corner_config['linear_perturbation.weight']=torch.Tensor(np.zeros((90*90, 1), dtype='float32'))
corner_config['linear_perturbation.bias']=torch.Tensor(np.zeros((90*90,), dtype='float32'))
model_corners.load_state_dict(corner_config)

test_df = pd.read_csv("detection/MNIST/test.csv")
testinggData = CustomMnistDataset_OL(test_df, test=True)

test_dataloader = DataLoader(testinggData, batch_size=1, shuffle=False)
test_iterator = iter(test_dataloader)

print("Let's apply it to test dataset:", len(test_dataloader))

list_info_time = []
images_exp = []
eps_list = np.linspace(0, 0.00560,100)


for image_id in range(3):
    list_info = []
    print(f"Begin to work with image {image_id}")
    st_im = time.time()
    data = next(test_iterator)
    X, y = data
    gt_logit, gt_box = y
    gt_box = gt_box.detach().numpy()[0]

    for i in range(len(eps_list)):
        start_perturbation = time.time() 
        lb_box_wn, ub_box_wn = bound_whitenoise(model_box, X, eps_list[i])
        lb_box_bri, ub_box_bri = bound_brightness(model_corners, X, eps_list[i])
        lb_box_contr, ub_box_contr = bound_brightness(model_corners, X, eps_list[i])
        end_perturbation = time.time()
        st_computed_ious = time.time()
        ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])

        perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn],
                               "brightness":[lb_box_bri, ub_box_bri], 
                               "contrast":[lb_box_contr, ub_box_contr]}
        for perturbation_name, bounds in perturbations_dict.items():
            print("perturbations_dict", perturbation_name, bounds)
            lb_box, ub_box = bounds[0], bounds[1]
            lb_box = [clip_corner(corner) for corner in lb_box]
            ub_box = [clip_corner(corner) for corner in ub_box]
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
        
            list_info.append(Merge({"image_id":image_id, 
                                "gt_logit": gt_logit.item(),
                                "eps":eps_list[i], 
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
    df.to_csv(f"results/many_perturbations/{image_id}_iou_calculations.csv")
  

pd.DataFrame(list_info_time).to_csv("results/many_perturbations/times2.csv")
pd.DataFrame(images_exp).to_csv("results/many_perturbations/images.csv")

