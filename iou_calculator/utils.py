from Interval import Interval
from Hyperrectangle import Hyperrectangle
from Hyperrectangle_interval import Hyperrectangle_interval
from IoU import IoU

def print_results(x_1=Interval(0,1), x_2=Interval(10,11), y_1=Interval(0,1), y_2=Interval(10,11), x_bl=0.5,x_tr=9.5, y_bl=0.5, y_tr=10.5):
    try:
        hyperrect_interval = Hyperrectangle_interval(x_1=x_1, x_2=x_2, y_1=y_1, y_2=y_2)
        hyperrect = Hyperrectangle(x_bl=x_bl,x_tr=x_tr, y_bl=y_bl, y_tr=y_tr)
        intersection = IoU(hyperrect_interval, hyperrect)
        intersection.plot()
        iou = intersection.iou()
        if not iou['IoU_vanilla_reluval'].contains(iou['IoU_optim']):
            raise ValueError("Optimized IoU should be included in vanilla IoU.")
            
        print("--------------- results ⧉---------------")
        intersection.iou(display = True)


        print("--------------- reluval method 📸 ---------------")
        intersection.iou_reluval(display = True)

        print("--------------- optimized method 📸 ---------------")
        intersection.iou_optim(display = True)
    
    except ValueError as e:
        print(e)