from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np


def bound_whitenoise(model_box, model_digit, X, eps):
    X_lirpa = X.float().to('cpu')

    mask = np.sign(X_lirpa[0])
    lb_ = 0.*X_lirpa
    ub_ = mask

    model_lirpa_corner = BoundedModule(model_box, X_lirpa)
    model_lirpa_digit = BoundedModule(model_digit, X_lirpa)

    x_min = X_lirpa - eps*mask
    x_max = X_lirpa + eps*mask

    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=x_min, x_U=x_max)
    input_lirpa = BoundedTensor(X_lirpa, ptb)

    lb_box, ub_box = model_lirpa_corner.compute_bounds(x=(input_lirpa,),IBP=True, method='crown')

        
    lb_adv, ub_adv = model_lirpa_digit.compute_bounds(x=(input_lirpa,), IBP=True, method='crown')

    #score_adv = ((ub_adv - lb_adv[0,5])[0,np.array([i for i in range(10) if i!=5])]).max()


    return lb_box.detach().numpy()[0], ub_box.detach().numpy()[0], lb_adv.detach().numpy()[0], ub_adv.detach().numpy()[0]

