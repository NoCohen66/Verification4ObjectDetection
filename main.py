from solver.model.LARD.LARD import Neural_network_LARD, Neural_network_LARD_BrightnessContrast
from solver.data.d_loc.custom_from_MNIST import CustomMnistDataset_OL
from solver.model.d_loc.d_loc import NeuralNetwork_OL_v2, NeuralNetwork_BrightnessContrast
from solver.perturbation import bound_whitenoise, bound_brightness, bound_contrast, bound_brightness_LARD, bound_contrast_LARD
from iou_calculator.bounding_box_utils.bounding_box_interval_values import Hyperrectangle_interval
from iou_calculator.bounding_box_utils.bounding_box import Hyperrectangle
from iou_calculator.bounding_box_utils.interval import Interval
from iou_calculator.IoU import IoU
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
parser.add_argument('-w', '--eps_list_whitenoise', default= np.linspace(0, 0.002,10), help="Range of variation for whitenoise perturbation.")
parser.add_argument('-b', '--eps_list_brightness', default= np.linspace(0, 0.002,10), help="Range of variation for brightness perturbation.")
parser.add_argument('-c', '--eps_list_contrast', default= np.linspace(0, 0.01,10), help="Range of variation for contrast perturbation.")
parser.add_argument('-m', '--methods_list', nargs="+", default=['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)'], help="Methods use to compute bounds.")
parser.add_argument('-nb','--nb_images', default=40, help="Quantity of images to be processed.")

parser.add_argument('--d_loc_model_weights', default='solver/model/d_loc/weights', help="Location of the regression model trained using the d_loc dataset.")
parser.add_argument('--d_loc_dataset', default="solver/data/d_loc/test.csv", help="Location of the d_loc test dataset.")
parser.add_argument('--d_loc_results_path', default='results/d_loc', help="Directory for storing the d_loc results.")

parser.add_argument('--LARD_model_weights', default='solver/model/LARD/weights', help="Location of the object detection model trained using the LARD dataset.")
parser.add_argument('--LARD_dataset', default='solver/data/LARD/test.pkl', help="Location of the LARD test dataset.")
parser.add_argument('--LARD_results_path', default='results/LARD', help="Directory for storing the LARD results.")
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

    if args.dataset_model == "d_loc":

        model_torch_load  = torch.jit.load(f'{args.d_loc_model_weights}.pt')
        model_box = NeuralNetwork_OL_v2(classif=False)
        model_box.load_state_dict(model_torch_load.state_dict())

    
      
        '''
        To apply brightness and contrast perturbations effectively, we utilize the perturbation coefficient, alpha. 
        For brightness adjustments, alpha is added, and for contrast, it is multiplied.
        The tool auto-lirpa is not compatible with neural networks designed to take alpha as the single input during their forward operation (usually design to take an image as input)
        To accommodate this, we have developed a neural network that depends on the image and includes a linear layer. 
        This layer is responsible for performing either the addition or multiplication of the image with alpha as the input.
        '''

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

        test_df = pd.read_csv(args.d_loc_dataset)
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

            print("d_loc model")
            print(args.methods_list)
            for method in list(args.methods_list):
                print("Method:", method)
                for i in range(len(eps_list_whitenoise)):
        
                    start_perturbation = time.time() 
                    lb_box_wn, ub_box_wn = bound_whitenoise(model_box, X, eps_list_whitenoise[i], method=method.split()[0])
                    lb_box_bri, ub_box_bri = bound_brightness(model_corners, X, eps_list_brightness[i], method=method.split()[0])
                    lb_box_contr, ub_box_contr = bound_contrast(model_corners_contrast, X, eps_list_contrast[i], method=method.split()[0])
                    end_perturbation = time.time()
                    st_computed_ious = time.time()
                    ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])

                    perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn],
                                        "brightness":[lb_box_bri, ub_box_bri], 
                                        "contrast":[lb_box_contr, ub_box_contr]}
                    
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
                                            "image_id":image_id, 
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
            df.to_csv(f"{args.d_loc_results_path}/{image_id}_iou_calculations.csv")
        

        pd.DataFrame(list_info_time).to_csv(f"{args.d_loc_results_path}/times.csv")
        pd.DataFrame(images_exp).to_csv(f"{args.d_loc_results_path}/images.csv")

    elif args.dataset_model == "LARD": 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        model_torch_load  = torch.load(f'{args.LARD_model_weights}.pt', torch.device('cpu')) # change if GPU
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

        X_ = dico_dataset['x_train']
        Y_ = dico_dataset['y_train']
        X_ = torch.Tensor(X_).to(device)
        Y_ = torch.Tensor(Y_).to(device)

        list_info_time = []
        images_exp = []


        for image_id in range(int(args.nb_images)):
            image_id = image_id + 1 #not dealing with image 0
            list_info = []
            print(f"Begin to work with image {image_id}")
            st_im = time.time()

            X = X_[-image_id][None]/255
            y = Y_[-image_id]*256
            gt_box = y
            gt_box = gt_box.detach().numpy()
            ground_truth_box = Hyperrectangle(x_bl=gt_box[0],x_tr=gt_box[2], y_bl=gt_box[1], y_tr=gt_box[3])
        
            for method in list(args.methods_list):
                print("Method:", method)
                for i in range(len(eps_list_contrast)):
                    start_perturbation = time.time() 
                    lb_box_contr, ub_box_contr = bound_contrast_LARD(model_contr, X, eps_list_contrast[i], method=method.split()[0])
                    lb_box_wn, ub_box_wn = bound_whitenoise(torch_model, X, eps_list_whitenoise[i], method=method.split()[0])
                    lb_box_bri, ub_box_bri = bound_brightness_LARD(model_bri, X, eps_list_brightness[i], method=method.split()[0])
                    end_perturbation = time.time()
                    st_computed_ious = time.time()
                    perturbations_dict = {"whitenoise": [lb_box_wn, ub_box_wn], 
                                        "brightness": [lb_box_bri, ub_box_bri], 
                                        "contrast":[lb_box_contr, ub_box_contr]}
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
                                            "image_id":image_id, 
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
            df.to_csv(f"{args.LARD_results_path}/{image_id}_iou_calculations.csv")
        

        pd.DataFrame(list_info_time).to_csv(f"{args.LARD_results_path}/times.csv")
        pd.DataFrame(images_exp).to_csv(f"{args.LARD_results_path}/images.csv")


    else:
        raise ValueError(f"Dataset {args.dataset_model} not recognized.")


if __name__ == "__main__":
    main()
