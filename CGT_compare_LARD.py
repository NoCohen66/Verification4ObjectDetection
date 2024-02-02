from solver.LARD.model.NeuralNetwork_LARD import Neural_network_LARD, Neural_network_LARD_BrightnessContrast
from solver.MNIST.model.NeuralNetwork_OL_v2 import NeuralNetwork_OL_v2
from solver.MNIST.data.CustomMnistDataset_OL import CustomMnistDataset_OL
from solver.MNIST.model.NeuralNetwork_BrightnessContrast import NeuralNetwork_BrightnessContrast
from solver.perturbation import bound_whitenoise, bound_brightness_LARD_input, bound_whitenoise_input, bound_brightness, bound_contrast, bound_brightness_LARD, bound_contrast_LARD
from iou_calculator.Hyperrectangle_interval import Hyperrectangle_interval
from iou_calculator.Hyperrectangle import Hyperrectangle
from iou_calculator.Interval import Interval
from iou_calculator.IoU import IoU
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from CGT_utils import IntervalTensor

from iou_calculator.utils import Merge, clip_corner
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

parser.add_argument('-d', '--dataset_model', default="LARD", help="The dataset and model to use.")
parser.add_argument('-w', '--eps_list_whitenoise', default= np.linspace(0.0001, 0.0006,11), help="Range of variation for whitenoise perturbation.")
parser.add_argument('-b','--eps_list_brightness', default= np.linspace(0.0001, 0.0006,11), help="Range of variation for brightness perturbation.")
parser.add_argument('-c','--eps_list_contrast', default= np.linspace(0.0001, 0.0006,11), help="Range of variation for contrast perturbation.")
#parser.add_argument('-m','--methods_list', nargs="+", default=['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)'], help="Methods use to compute bounds.")
parser.add_argument('-m','--methods_list', nargs="+", default=['IBP+backward (CROWN-IBP)'], help="Methods use to compute bounds.")

parser.add_argument('-nb','--nb_images', default=2, help="Quantity of images to be processed.")

parser.add_argument('--LARD_model_torch_load_filename', default='solver/LARD/model/tmp_nfm_v4', help="Location of the object detection model trained using the LARD dataset.")
parser.add_argument('--LARD_dataset', default='solver/LARD/data/lard_nfm_data_iou.pkl', help="Location of the LARD test dataset.")
parser.add_argument('--LARD_results_path', default='results/LARD', help="Directory for storing the LARD results.")
args = parser.parse_args()

eps_list_whitenoise = args.eps_list_whitenoise
eps_list_brightness = args.eps_list_brightness
eps_list_contrast = args.eps_list_contrast
pertubations_values = {"whitenoise":eps_list_whitenoise, 
                        "brightness": eps_list_brightness, 
                        #"contrast": eps_list_contrast}
                        }
if not len(eps_list_whitenoise) == len(eps_list_brightness) == len(eps_list_contrast):
    raise ValueError("Perturbation ranges should be the same.")


def main(): 

    if args.dataset_model == "LARD": 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        model_torch_load  = torch.load(f'{args.LARD_model_torch_load_filename}.pt', torch.device('cpu')) # change if GPU
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


        with closing((open(args.LARD_dataset, 'rb'))) as f:
                    dico_dataset = pkl.load(f)

        X_train = dico_dataset['x_train']
        X_test = dico_dataset['x_test']
        Y_train = dico_dataset['y_train']
        Y_test = dico_dataset['y_test']
        X_train_ = torch.Tensor(X_train).to(device)
        X_test_ = torch.Tensor(X_test).to(device)
        Y_train_ = torch.Tensor(Y_train).to(device)
        Y_test_ = torch.Tensor(Y_test).to(device)

        list_info_time = []
        images_exp = []


        for image_id in range(int(args.nb_images)):
            image_id = image_id + 1 #not dealing with image 0
            list_info = []
            print(f"Begin to work with image {image_id}")
            st_im = time.time()

            X = X_train_[-image_id][None]/255
            y = Y_train_[-image_id]*256
            gt_box = y
            gt_box = gt_box.detach().numpy()
            ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])
        
            for method in list(args.methods_list):
                print("Method:", method)
                for i in range(len(eps_list_contrast)):
                    print(eps_list_whitenoise[i])
                    start_perturbation = time.time() 
                    #lb_box_contr, ub_box_contr = bound_contrast_LARD(model_contr, X, eps_list_contrast[i], method=method.split()[0])
                    lb_box_wn, ub_box_wn = bound_whitenoise(torch_model, X, eps_list_whitenoise[i], method=method.split()[0])
                    lb_box_bri, ub_box_bri = bound_brightness_LARD(model_bri, X, eps_list_brightness[i], method=method.split()[0])
                    input_whitenoise = bound_whitenoise_input(torch_model, X, eps_list_whitenoise[i], method=method.split()[0])
                    input_brightness = bound_brightness_LARD_input(torch_model, X, eps_list_whitenoise[i], method=method.split()[0])

                    
                    # CGT
                    X_lirpa = X.float().to('cpu')
                    ibp_net = BoundedModule(torch_model, X_lirpa)
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
                    
                    if torch.all(bounded_inputs == input_whitenoise):
                         print("CGT is equal to input_whitenoise")
                    else:
                         print("CGT is NOT equal to input_whitenoise")

                    if torch.all(bounded_inputs == input_brightness):
                         print("CGT is equal to brightness")
                    else:
                         print("CGT is NOT equal to brightness ")
                    
                    

                    # CGT not clamp
                    X_lirpa = X.float().to('cpu')
                    ibp_net = BoundedModule(torch_model, X_lirpa)
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
                    
                    
                    
                    perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn], 
                                        "brightness": [lb_box_bri, ub_box_bri], 
                                        #"contrast":[lb_box_contr, ub_box_contr], 
                                        "CGT_brightness":[lb_CGT.detach().numpy()[0], ub_CGT.detach().numpy()[0]], 
                                        "CGT_brightness_clamp": [lb_CGT_clamp.detach().numpy()[0], ub_CGT_clamp.detach().numpy()[0]]}
                    for perturbation_name, bounds in perturbations_dict.items():
                        eps_list = pertubations_values[perturbation_name]
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
            df.to_csv(f"CGT_comp_LARD/{image_id}_iou_calculations.csv")
        

        pd.DataFrame(list_info_time).to_csv(f"CGT_comp_LARD/times.csv")
        pd.DataFrame(images_exp).to_csv(f"CGT_comp_LARD/images.csv")


    else:
        raise ValueError(f"Dataset {args.dataset_model} not recognized.")


if __name__ == "__main__":
    main()