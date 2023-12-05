from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import torch 
import pandas as pd
import torch.nn.functional as F
from torch import nn



def bound_whitenoise(model_box, X, eps, method='crown'):
    X_lirpa = X.float().to('cpu')
    model_lirpa_corner = BoundedModule(model_box, X_lirpa)
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    input_lirpa = BoundedTensor(X_lirpa, ptb)
    lb_box, ub_box = model_lirpa_corner.compute_bounds(x=(input_lirpa,),method=method)
    return lb_box.detach().numpy()[0], ub_box.detach().numpy()[0]


def set_brightness(model, image):
    # create a pytorch model for a specific flatten image
    # once a model has been set for one images, you can call it for any brightness value
    image_flatten = image.view((90*90,))
    brightness_config = model.state_dict()
    brightness_config['linear_perturbation.bias']=image_flatten
    brightness_config['linear_perturbation.weight']=torch.Tensor(np.ones((90*90,1), dtype='float32'))
    model.load_state_dict(brightness_config)

def set_contrast(model, image):
    # create a pytorch model for a specific flatten image
    # once a model has been set for one images, you can call it for any contrast value
    image_flatten = image.view((90*90,1))
    contrast_config = model.state_dict()
    contrast_config['linear_perturbation.bias']=torch.Tensor(np.zeros((90*90,), dtype='float32'))
    contrast_config['linear_perturbation.weight']=image_flatten
    model.load_state_dict(contrast_config)



def bound_contrast(model_corners, X, brightness_variations, method='crown'):
    tensor_init_contrast = torch.tensor([[1.0]]).float().to('cpu')
    set_contrast(model_corners, X)
    model_lirpa_corners = BoundedModule(model_corners, tensor_init_contrast)
    ptb_brightness = PerturbationLpNorm(norm=np.inf, eps=brightness_variations) 
    input_lirpa_brightness = BoundedTensor(tensor_init_contrast, ptb_brightness)
    lb_brightness, ub_brightness = model_lirpa_corners.compute_bounds(x=(input_lirpa_brightness,),
                                                        method=method)
    return lb_brightness.detach().numpy()[0], ub_brightness.detach().numpy()[0]

def bound_brightness(model_corners, X, eps, method='crown'):
    tensor_init_brightness = torch.tensor([[0.0]]).float().to('cpu')
    set_brightness(model_corners, X)
    model_lirpa_corners = BoundedModule(model_corners, tensor_init_brightness)
    ptb_brightness = PerturbationLpNorm(norm=np.inf, eps=eps) 
    input_lirpa_brightness = BoundedTensor(tensor_init_brightness, ptb_brightness)
    lb_brightness, ub_brightness = model_lirpa_corners.compute_bounds(x=(input_lirpa_brightness,),
                                                        method=method)
    # lb_brightness, ub_brightness = model_lirpa_corners.compute_bounds(x=(input_lirpa_brightness,),method='backward')
    return lb_brightness.detach().numpy()[0], ub_brightness.detach().numpy()[0]


def set_brightness_LARD(model, image):
    # create a pytorch model for a specific flatten image
    # once a model has been set for one images, you can call it for any brightness value
    image_flatten = image.view((3*256*256,)) # mean on grey scale :/
    brightness_config = model.state_dict()
    brightness_config['linear_perturbation.bias']=image_flatten
    brightness_config['linear_perturbation.weight']=torch.Tensor(np.ones((3*256*256,1), dtype='float32'))
    model.load_state_dict(brightness_config)

def set_contrast_LARD(model, image):
    # create a pytorch model for a specific flatten image
    # once a model has been set for one images, you can call it for any contrast value
    image_flatten = image.view((3*256*256,1))
    contrast_config = model.state_dict()
    contrast_config['linear_perturbation.bias']=torch.Tensor(np.zeros((3*256*256,), dtype='float32'))
    contrast_config['linear_perturbation.weight']=image_flatten
    model.load_state_dict(contrast_config)

def bound_brightness_LARD(model_corners, X, eps, method='crown'):
    tensor_init_brightness = torch.tensor([[0.0]]).float().to('cpu')
    set_brightness_LARD(model_corners, X)
    model_lirpa_corners = BoundedModule(model_corners, tensor_init_brightness)
    ptb_brightness = PerturbationLpNorm(norm=np.inf, eps=eps) 
    input_lirpa_brightness = BoundedTensor(tensor_init_brightness, ptb_brightness)
    lb_brightness, ub_brightness = model_lirpa_corners.compute_bounds(x=(input_lirpa_brightness,),
                                                        method=method)
    return lb_brightness.detach().numpy()[0], ub_brightness.detach().numpy()[0]

def bound_contrast_LARD(model_corners, X, brightness_variations, method='crown'):
    tensor_init_contrast = torch.tensor([[1.0]]).float().to('cpu')
    set_contrast_LARD(model_corners, X)
    model_lirpa_corners = BoundedModule(model_corners, tensor_init_contrast)
    ptb_brightness = PerturbationLpNorm(norm=np.inf, eps=brightness_variations) 
    input_lirpa_brightness = BoundedTensor(tensor_init_contrast, ptb_brightness)
    lb_brightness, ub_brightness = model_lirpa_corners.compute_bounds(x=(input_lirpa_brightness,),
                                                        method=method)
    return lb_brightness.detach().numpy()[0], ub_brightness.detach().numpy()[0]
