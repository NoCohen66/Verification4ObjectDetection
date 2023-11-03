from Interval import Interval
from Hyperrectangle import Hyperrectangle
from Hyperrectangle_interval import Hyperrectangle_interval
from IoU import IoU
import pandas as pd
from utils import print_results
import json
import time

hyperrect_interval = Hyperrectangle_interval(x_1=Interval(-3,-2), x_2=Interval(-1,0), y_1=Interval(-4,-3), y_2=Interval(-2,-1))
kk = pd.DataFrame({"kk":hyperrect_interval}, index=[0])
kk.to_csv("test.csv")
df = pd.read_csv("test.csv")
json.loads(df["kk"][0])
#print(type(df[0]))
#df = pd.read_csv("../detection/bound_results/results.csv")
#print(type(df.lb_box[0]))
"""
df.lb_box = df.lb_box.apply(json.loads)
df.ub_box = df.ub_box.apply(json.loads)
st = time.time()
df['hyperrect_interval'] = df.apply(lambda x: Hyperrectangle_interval(x_1=Interval(x.lb_box[0],x.ub_box[0]), x_2=Interval(x.lb_box[2],x.ub_box[2]), y_1=Interval(x.lb_box[1],x.ub_box[1]), y_2=Interval(x.lb_box[3],x.ub_box[3])), axis=1)
et = time.time()
print("elapsed_time for 4 images", et - st)
"""