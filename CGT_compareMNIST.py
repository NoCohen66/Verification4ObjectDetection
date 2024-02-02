from solver.LARD.model.NeuralNetwork_LARD import Neural_network_LARD, Neural_network_LARD_BrightnessContrast
from solver.MNIST.model.NeuralNetwork_OL_v2 import NeuralNetwork_OL_v2
from solver.MNIST.data.CustomMnistDataset_OL import CustomMnistDataset_OL
from solver.MNIST.model.NeuralNetwork_BrightnessContrast import NeuralNetwork_BrightnessContrast
from solver.perturbation import bound_whitenoise, bound_whitenoise_xLxU, bound_brightness, bound_contrast, bound_brightness_LARD, bound_contrast_LARD
from iou_calculator.Hyperrectangle_interval import Hyperrectangle_interval
from iou_calculator.Hyperrectangle import Hyperrectangle
from iou_calculator.Interval import Interval
from iou_calculator.IoU import IoU
from iou_calculator.utils import Merge, clip_corner
from CGT_utils import IntervalTensor
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from IPython.display import display
import torch 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time 
import argparse
from contextlib import closing
import pickle as pkl

parser = argparse.ArgumentParser(description="Process some datasets and networks.")

parser.add_argument('-d', '--dataset_model', default="MNIST", help="The dataset and model to use.")
parser.add_argument('-w', '--eps_list_whitenoise', default= np.linspace(0, 0.002,11), help="Range of variation for whitenoise perturbation.")
parser.add_argument('-b','--eps_list_brightness', default= np.linspace(0, 0.002,11), help="Range of variation for brightness perturbation.")
parser.add_argument('-c','--eps_list_contrast', default= np.linspace(0, 0.001,11), help="Range of variation for contrast perturbation.")
#parser.add_argument('-m','--methods_list', nargs="+", default=['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)'], help="Methods use to compute bounds.")
parser.add_argument('-m','--methods_list', nargs="+", default=['IBP+backward (CROWN-IBP)'], help="Methods use to compute bounds.")
parser.add_argument('-nb','--nb_images', default=2, help="Quantity of images to be processed.")

parser.add_argument('--MNIST_model_corner_filename', default='solver/MNIST/model/toy_model_corners', help="Location of the regression model trained using the MNIST dataset.")
parser.add_argument('--MNIST_dataset', default="solver/MNIST/data/test.csv", help="Location of the MNIST test dataset.")
parser.add_argument('--MNIST_results_path', default='results/MNIST', help="Directory for storing the MNIST results.")
args = parser.parse_args()

eps_list_whitenoise = args.eps_list_whitenoise
eps_list_brightness = args.eps_list_brightness
eps_list_contrast = args.eps_list_contrast
pertubations_values = {"whitenoise":eps_list_whitenoise, 
                        "brightness": eps_list_brightness, 
                        "contrast": eps_list_contrast}
if not len(eps_list_whitenoise) == len(eps_list_brightness) == len(eps_list_contrast):
    raise ValueError("Perturbation ranges should be the same.")


def main(): 

    if args.dataset_model == "MNIST":

        model_torch_load  = torch.jit.load(f'{args.MNIST_model_corner_filename}.pt')
        model_box = NeuralNetwork_OL_v2(classif=False)
        model_box.load_state_dict(model_torch_load.state_dict())

        model_corners = NeuralNetwork_BrightnessContrast(classif=False)
        corner_config = model_torch_load.state_dict()
        corner_config['linear_perturbation.weight']=torch.Tensor(np.zeros((90*90, 1), dtype='float32'))
        corner_config['linear_perturbation.bias']=torch.Tensor(np.zeros((90*90,), dtype='float32'))
        model_corners.load_state_dict(corner_config)

        model_corners_contrast = NeuralNetwork_BrightnessContrast(classif=False)
        corner_config_contrast = model_torch_load.state_dict()
        corner_config_contrast['linear_perturbation.weight']=torch.Tensor(np.zeros((90*90, 1), dtype='float32'))
        corner_config_contrast['linear_perturbation.bias']=torch.Tensor(np.zeros((90*90,), dtype='float32'))
        model_corners_contrast.load_state_dict(corner_config_contrast)

        test_df = pd.read_csv(args.MNIST_dataset)
        testinggData = CustomMnistDataset_OL(test_df, test=True)
        test_dataloader = DataLoader(testinggData, batch_size=1, shuffle=True)
        test_iterator = iter(test_dataloader)

        list_info_time = []
        images_exp = []
        
        
        for image_id in range(int(args.nb_images)):
            list_info = []
            print(f"Begin to work with image {image_id}")
            st_im = time.time()
            data = next(test_iterator)
            X, y = data
            gt_logit, gt_box = y
            gt_box = gt_box.detach().numpy()[0]

            for method in list(args.methods_list):
                print("Method:", method)
                for i in range(len(eps_list_whitenoise)):
        
                    lb_box_wn, ub_box_wn = bound_whitenoise(model_box, X, eps_list_whitenoise[i], method=method.split()[0])
                    
                    lb_box_bri, ub_box_bri = bound_brightness(model_corners, X, eps_list_brightness[i], method=method.split()[0])
                    #lb_box_contr, ub_box_contr = bound_contrast(model_corners_contrast, X, eps_list_contrast[i], method=method.split()[0])
                    
                    # CGT
                    X_lirpa = X.float().to('cpu')
                    ibp_net = BoundedModule(model_box, X_lirpa)
                    interval_inputs = IntervalTensor(X_lirpa, X_lirpa)
                    theta_min, theta_max = -eps_list_brightness[i], eps_list_brightness[i]
                    #interval_inputs = IntervalTensor((interval_inputs.lower + theta_min).clamp(0, 1), (interval_inputs.upper + theta_max).clamp(0, 1))
                    interval_inputs = IntervalTensor((interval_inputs.lower + theta_min), (interval_inputs.upper + theta_max))
                    inputs_L = interval_inputs.lower
                    inputs_U = interval_inputs.upper
                    ptb = PerturbationLpNorm(norm=np.inf, x_L=inputs_L, x_U=inputs_U)
                    bounded_inputs = BoundedTensor(X_lirpa, ptb) 
                    lb_CGT, ub_CGT = ibp_net(method_opt="compute_bounds", x=(bounded_inputs,), method=method.split()[0])
                    pertubations_values["CGT_brightness"] = [eps_list_brightness[i]]*len(eps_list_whitenoise)

                    # CGT not clamp
                    X_lirpa = X.float().to('cpu')
                    ibp_net = BoundedModule(model_box, X_lirpa)
                    interval_inputs = IntervalTensor(X_lirpa, X_lirpa)
                    theta_min, theta_max = -eps_list_brightness[i], eps_list_brightness[i]
                    interval_inputs_clamp = IntervalTensor((interval_inputs.lower + theta_min).clamp(0, 1), (interval_inputs.upper + theta_max).clamp(0, 1))
                    #interval_inputs = IntervalTensor((interval_inputs.lower + theta_min), (interval_inputs.upper + theta_max))
                    inputs_L_clamp = interval_inputs_clamp.lower
                    inputs_U_clamp = interval_inputs_clamp.upper
                    ptb_clamp = PerturbationLpNorm(norm=np.inf, x_L=inputs_L_clamp, x_U=inputs_U_clamp)
                    bounded_inputs_clamp = BoundedTensor(X_lirpa, ptb_clamp) 
                    lb_CGT_clamp, ub_CGT_clamp = ibp_net(method_opt="compute_bounds", x=(bounded_inputs_clamp,), method=method.split()[0])
                    pertubations_values["CGT_brightness_clamp"] = [eps_list_brightness[i]]*len(eps_list_whitenoise)           
                    
                    # Comparison whitenoise
                    lb_box_wn_xLxU, ub_box_wn_xLxU = bound_whitenoise_xLxU(model_box, X, inputs_L,inputs_U, method=method.split()[0])
                    pertubations_values["whitenoise_xLxU"] = [eps_list_brightness[i]]*len(eps_list_whitenoise)
                  
                    ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])

                    perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn],
                                          "whitenoise_xLxU": [lb_box_wn_xLxU, ub_box_wn_xLxU],
                                        "brightness":[lb_box_bri, ub_box_bri], 
                                        "CGT_brightness_clamp":[lb_CGT_clamp.detach().numpy()[0], ub_CGT_clamp.detach().numpy()[0]], 
                                        "CGT_brightness":[lb_CGT.detach().numpy()[0], ub_CGT.detach().numpy()[0]]}
                    
                    #for name, bounds in perturbations_dict.items():
                     #   print(name, bounds)
                  
                    for perturbation_name, bounds in perturbations_dict.items():
                        eps_list = pertubations_values[perturbation_name]
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
                    
                        list_info.append(Merge({"method":method,
                                            "eps":eps_list[i],
                                            "fake_iou": fake_iou,
                                            "perturbation":perturbation_name, 
                                            "lb_box":lb_box,
                                            "ub_box": ub_box}, dict_iou))

                

        
            et_im = time.time()
            print(f"Time to proceed one image {et_im-st_im}")
            list_info_time.append((image_id, et_im-st_im))
            images_exp.append(X[0,0,:,:].flatten().tolist())
            df = pd.DataFrame(list_info)
            df.to_csv(f"CGT_comp/{image_id}_iou_calculations.csv")
        

        pd.DataFrame(list_info_time).to_csv(f"CGT_comp/times.csv")
        pd.DataFrame(images_exp).to_csv(f"CGT_comp/images.csv")

    else:
        print("For this experimentation the model is MNIST.")

if __name__ == "__main__":
    main()