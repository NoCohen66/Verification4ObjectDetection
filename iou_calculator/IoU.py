from iou_calculator.bounding_box_utils.interval import Interval
from iou_calculator.bounding_box_utils.bounding_box import Hyperrectangle
from iou_calculator.bounding_box_utils.bounding_box_interval_values import Hyperrectangle_interval
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
import time


class IoU:
    """Class representing the intersection between an hyperrectangle and an hyperrectangle_interval."""
    
    def __init__(self, hyperrectangle_interval, hyperrectangle):
        """
        Constructor of the Intersection_between_hyperrectangle_interval_and_hyperrectangle_onx1 class.
        
        hyperrectangle_interval: (hyperrectangle_interval) hyperrectangle where x1 coordinate of the rectangle have an interval of variation.
        hyperrectangle:           (hyperrectangle) hyperrectangle
        """
        
        if type(hyperrectangle_interval) != Hyperrectangle_interval or type(hyperrectangle) != Hyperrectangle:
            raise ValueError("The type of the inputs are not  Hyperrectangle_interval or Hyperrectangle.") 

        
        self.hyperrect_interval = hyperrectangle_interval
        self.hyperrect = hyperrectangle


        if self.hyperrect.area() <= 0:
            raise ValueError("To use IoU method, the ground truth should have a positive (not zero) area.") 

        # Information about their interactions
        four_points = self.hyperrect.four_points()
        df = pd.DataFrame(four_points, index=["xaxis", "yaxis"]).T
        df["is_inside_max_area"] = df.apply(lambda x: self.hyperrect_interval.is_inside_max_area([x.xaxis, x.yaxis]), axis=1)
        df["is_inside_min_area"] = df.apply(lambda x: self.hyperrect_interval.is_inside_min_area([x.xaxis, x.yaxis]), axis=1)
        self.df = df


        
    def chech_case_x1(self):
        """
        This function show in which case with respect to the latex document 

        Explanation for different conditions:
        
         1. Case A (the two rectangle do not overlap at all): 
            C1 - max(x1_p, x1_gt) <= 0
        
         2. Case B: 
            C1 - max(x1_p, x1_gt) > 0 AND
            x1_gt >= x1_p

         3. Case C:
            C1 - max(x1_p, x1_gt) > 0 AND
            x1_gt < x1_p

        This uses the function "is_inside" that check if a point is in a rectangle or not.

        """
        four_points = self.hyperrect.four_points()
        df = pd.DataFrame(four_points, index=["xaxis", "yaxis"]).T
        df["is_inside_max_area"] = df.apply(lambda x: self.hyperrect_interval.is_inside_max_area([x.xaxis, x.yaxis]), axis=1)
        df["is_inside_min_area"] = df.apply(lambda x: self.hyperrect_interval.is_inside_min_area([x.xaxis, x.yaxis]), axis=1)
        
        if df["is_inside_max_area"].sum() == 0:
             return("case A")
        else:
            if Interval(self.hyperrect.x_bl) >= self.hyperrect_interval.x_1:
                return("case B")
            else: 
                return("case C")

            
    def iou_simple(self):
        if self.hyperrect_interval.min_hyperrect().dict()==self.hyperrect_interval.max_hyperrect().dict():
            box = self.hyperrect_interval.min_hyperrect()
            overlap = box.overlap(self.hyperrect)
            return(overlap/(box.area()+ self.hyperrect.area()-overlap))


        else: 
            raise ValueError("The predicted box have coordinates within an interval, you cannot use simple iou.")
  
    def create_interval(self,value1, value2): 
        if value2 >= value1:
            return(Interval(value1, value2))
        elif value2 <= value1: 
            return(Interval(value2, value1))
        else:
            raise ValueError("To create an interval values should be comparable.")

    def choice(self, rect_choose, coord_gt_tuple, coord_p_name = "x_bl"):
        coord_gt, point = coord_gt_tuple 
        
        hyperrect_interval_min = self.hyperrect_interval.min_hyperrect()
        hyperrect_interval_max = self.hyperrect_interval.max_hyperrect()

        interval = self.create_interval(hyperrect_interval_max.dict()[coord_p_name], hyperrect_interval_min.dict()[coord_p_name])

        if interval.contains(coord_gt):
            rect_choose[coord_p_name] = coord_gt
        else:
            coord_p_min = hyperrect_interval_min.dict()[coord_p_name]
            coord_p_max = hyperrect_interval_max.dict()[coord_p_name]
            
            distance_withmin_x = abs(coord_gt - coord_p_min)
            distance_withmax_x = abs(coord_gt - coord_p_max)
            
            if distance_withmin_x <= distance_withmax_x:
                rect_choose[coord_p_name] = coord_p_min
            else: 
                rect_choose[coord_p_name] = coord_p_max
        return(rect_choose)

    def optim_rec(self):

        if self.df["is_inside_max_area"].sum() == 0:
            raise ValueError("There is no overlap between the two rectangles.") 
        else: 
            rect_choose = {} 
            bottomleft_gt = self.hyperrect.four_points()["bottom_left"]
            topright_gt   = self.hyperrect.four_points()["top_right"]

            x_bl_gt, y_bl_gt = bottomleft_gt
            x_tr_gt, y_tr_gt = topright_gt
            
            p_2_gt = {"x_bl": (x_bl_gt, bottomleft_gt), 
                      "y_bl": (y_bl_gt, bottomleft_gt), 
                      "x_tr": (x_tr_gt, topright_gt),
                      "y_tr": (y_tr_gt, topright_gt)}

            for key, value in p_2_gt.items():
                if self.hyperrect_interval.dict()[key].one_value():
                    rect_choose[key] = self.hyperrect_interval.dict()[key].one_value(returnV=True)
                    
                else:
                    rect_choose = self.choice(rect_choose, coord_gt_tuple = value, coord_p_name = key)


            
            optim_rec = Hyperrectangle(x_bl = rect_choose["x_bl"], 
                                                x_tr = rect_choose["x_tr"], 
                                                y_bl = rect_choose["y_bl"], 
                                                y_tr = rect_choose["y_tr"])
            
            if not self.hyperrect_interval.is_inside_variation_bl(optim_rec.four_points()["bottom_left"]):
                raise ValueError("The bottom left point of the optimization rectangle should be in the variation rectangle.")

            if not self.hyperrect_interval.is_inside_variation_tr(optim_rec.four_points()["top_right"]):
                raise ValueError("The top right point of the optimization rectangle should be in the variation rectangle.")
            return(optim_rec)



    def overlap_reluval(self):
        """
        Calculate the area of overlap between the two hyperrectangles, by calculating width and height.
 

        output: 
        (Interval) area of overlap

        """
        
        # Check if provided hyperrectangle is instance of the Hyperrectangle class
        if not isinstance(self.hyperrect, Hyperrectangle):
            raise ValueError("self.hyperrect should be instance of the Hyperrectangle class.")
        # Check if provided hyperrectangle is instance of the Hyperrectangle class
        if not isinstance(self.hyperrect_interval, Hyperrectangle_interval):
            raise ValueError("self.hyperrect_interval should be instance of the Hyperrectangle_interval class.")

        # Points for hyperrect_interval (predicted) that fit with latex description
        x_1_p = self.hyperrect_interval.x_1
        x_2_p = self.hyperrect_interval.x_2
        y_1_p = self.hyperrect_interval.y_1
        y_2_p = self.hyperrect_interval.y_2

        # Points for hyperrect_2 (ground truth) that fit with latex description
        x_1_gt = self.hyperrect.x_bl
        x_2_gt = self.hyperrect.x_tr
        y_1_gt = self.hyperrect.y_bl
        y_2_gt = self.hyperrect.y_tr

        """
        Calculate the width overlap between the two hyperrectangles.
        w_overlap = min(x_2_p, x_2_gt) - max(x_1_p, x_1_gt)
        w_overlap = alpha(x_2_p) - gamma(x_1_p)

        alpha(x_2_p) = x_2_p + x_2_gt - max(x_2_p, x_2_gt)
        gamma(x_1_p) = max(x_1_p, x_1_gt)
        """
        alpha_x_2_p = x_2_p.min_reluval(x_2_gt)
        gamma_x_1_p = x_1_p.max_reluval(x_1_gt)
        w_overlap = alpha_x_2_p - gamma_x_1_p
        
        """
        Calculate the height overlap between the two hyperrectangles.
        h_overlap = min(y_2_p, y_2_gt) - max(y_1_p, y_1_gt)
        h_overlap = alpha_y(y_2_p) - gamma_y(y_1_p)

        alpha_y(y_2_p) = y_2_p + y_2_gt - max(y_2_p, y_2_gt)
        gamma_y(y_1_p) = max(y_1_p, y_1_gt)        
        
        """
        alpha_y_2_p = y_2_p.min_reluval(y_2_gt)
        gamma_y_1_p = y_1_p.max_reluval(y_1_gt)
        h_overlap = alpha_y_2_p - gamma_y_1_p     

        """
        Calculate the area of overlap between the two hyperrectangles.
        
        Formula derived from:
        A_overlap = w_overlap * h_overlap
        """

        if w_overlap < Interval(0,0) or h_overlap < Interval(0,0):
            raise ValueError("width and height should be positive values.")

        return w_overlap * h_overlap

    def union_reluval(self):
        """
        Calculation for the area of union between the predicted bounding box and the ground truth:
        A_Union(x1_p, x2_p, y1_p, y2_p) =
                    A_p(x1_p, x2_p, y1_p, y2_p) + A_gt - A_Overlap(x1_p, x2_p, y1_p, y2_p) #
       
        For the specific scenario where we're focusing on the variation in x1_p:
        
        Let l_x1_p and u_x1_p be the concrete lower and upper bounds of x1_p respectively, such that l_x1_p, u_x1_p ∈ ℝ or {-∞, +∞}.
        
        We express x1_p as: x1_p = [l_x1_p, u_x1_p]
        
        Then, the area of union based on this interval is given by: #
        
        A_Union([l_x1_p, u_x1_p]) =
                                [A_p(u_x1_p) + A_gt - A_Overlap(l_x1_p), A_p(l_x1_p) + A_gt - A_Overlap(u_x1_p)]
        
        As A_Overlap, A_gt, A_p are already express as interval, the A_Union calculation is an interval addition.

        """
        A_gt  = Interval(self.hyperrect.area())
        A_p = self.hyperrect_interval.area()  
        A_overlap = self.overlap_reluval()
        
        return(A_p + A_gt - A_overlap)
    

    def iou_optim_max(self, display = False):
        # optim IBP
        A_gt  = self.hyperrect.area()
        optim_rec = self.optim_rec()
        A_overlap_optim = self.hyperrect.overlap(optim_rec)
        A_p_optim = optim_rec.area()
        A_union_optim = A_p_optim + A_gt - A_overlap_optim
        IoU_optim = A_overlap_optim/A_union_optim

        
        if display == False: 
            return(IoU_optim)
        elif display == True: 
            for key, values in dict_iou_optim.items(): 
                print(key, values)
        else: 
            dict_iou_optim = {"A_gt":A_gt, 
                "A_overlap": A_overlap_optim,
                "A_p":A_p_optim, 
                "A_union":A_union_optim, 
                "IoU":IoU_optim}
            return(dict_iou_optim)

    def contains_gt_list(self, interval, value):

        if interval.contains(value):
            return([interval.l, value, interval.u])
        else:
            return([interval.l, interval.u])


    def iou_optim_greedy(self, returnDf = False):
        A_gt  = self.hyperrect.area()
        
        # Points for hyperrect_interval (predicted) that fit with latex description
        x_1_p = self.hyperrect_interval.x_1
        x_2_p = self.hyperrect_interval.x_2
        y_1_p = self.hyperrect_interval.y_1
        y_2_p = self.hyperrect_interval.y_2

        # Points for hyperrect_2 (ground truth) that fit with latex description
        x_1_gt = self.hyperrect.x_bl
        x_2_gt = self.hyperrect.x_tr
        y_1_gt = self.hyperrect.y_bl
        y_2_gt = self.hyperrect.y_tr

        dico_mins = {"x_bl":[], "x_tr":[], "y_bl": [], "y_tr":[], "overlap": [], "iou":[]}



    
        for x_1_p_i in self.contains_gt_list(x_1_p, x_1_gt):
            for y_1_p_i in self.contains_gt_list(y_1_p, y_1_gt):
                for x_2_p_i in self.contains_gt_list(x_2_p, x_2_gt):
                    for y_2_p_i in self.contains_gt_list(y_2_p, y_2_gt):
                        #print(x_1_p_i, y_1_p_i, x_2_p_i, y_2_p_i)
                        dico_mins["x_bl"].append(x_1_p_i)
                        dico_mins["y_bl"].append(y_1_p_i)
                        dico_mins["x_tr"].append(x_2_p_i)
                        dico_mins["y_tr"].append(y_2_p_i)
                        hyp = Hyperrectangle(x_bl=x_1_p_i,
                                                      x_tr=x_2_p_i, 
                                                      y_bl=y_1_p_i, 
                                                      y_tr=y_2_p_i)
                        
                        A_p = hyp.area()
                        A_overlap = self.hyperrect.overlap(hyp)
                        dico_mins["overlap"].append(A_overlap)
                        A_union = A_p + A_gt - A_overlap
                        dico_mins["iou"].append(A_overlap/A_union)

        
        df = pd.DataFrame(dico_mins)

        if returnDf == False: 
            return([df.describe()["iou"]["min"], df.describe()["iou"]["max"]])
        else: 
            return(df)


    def iou_optim(self, returnDf = False):
            A_gt  = self.hyperrect.area()
            
            # Points for hyperrect_interval (predicted) that fit with latex description
            x_1_p = self.hyperrect_interval.x_1
            x_2_p = self.hyperrect_interval.x_2
            y_1_p = self.hyperrect_interval.y_1
            y_2_p = self.hyperrect_interval.y_2

            # Points for hyperrect_2 (ground truth) that fit with latex description
            x_1_gt = self.hyperrect.x_bl
            x_2_gt = self.hyperrect.x_tr
            y_1_gt = self.hyperrect.y_bl
            y_2_gt = self.hyperrect.y_tr

            dico_mins = {"x_bl":[], "x_tr":[], "y_bl": [], "y_tr":[], "overlap": [], "iou":[]}



        
            for x_1_p_i in [x_1_p.l, x_1_p.u]:
                for y_1_p_i in [y_1_p.l, y_1_p.u]:
                    for x_2_p_i in [x_2_p.l, x_2_p.u]:
                        for y_2_p_i in [y_2_p.l, y_2_p.u]:
                            #print(x_1_p_i, y_1_p_i, x_2_p_i, y_2_p_i)
                            dico_mins["x_bl"].append(x_1_p_i)
                            dico_mins["y_bl"].append(y_1_p_i)
                            dico_mins["x_tr"].append(x_2_p_i)
                            dico_mins["y_tr"].append(y_2_p_i)
                            hyp = Hyperrectangle(x_bl=x_1_p_i,
                                                        x_tr=x_2_p_i, 
                                                        y_bl=y_1_p_i, 
                                                        y_tr=y_2_p_i)
                            
                            A_p = hyp.area()
                     
                            A_overlap = self.hyperrect.overlap(hyp)
                            dico_mins["overlap"].append(A_overlap)
                            A_union = A_p + A_gt - A_overlap
                            dico_mins["iou"].append(A_overlap/A_union)

            
            df = pd.DataFrame(dico_mins)

            if returnDf == False: 
                return([df.describe()["iou"]["min"], self.iou_optim_max(display = False)])
            else: 
                return(df)


    def iou_reluval(self, display = False):
        """
        Definition of the Intersection over Union (IoU): 
            IoU = A_Overlap / A_Union 

        Example formula of IoU: Range of possible IoU values as x1_p varies within its bounds.
            
        When dealing with intervals, and considering that all area values are positive, i.e. for any l_x1_p, u_x1_p in ℝ^+* (positive reals excluding zero): 
            IoU([l_x1_p, u_x1_p]) =
            [A_Overlap(u_x1_p) / (A_p(l_x1_p) + A_gt - A_Overlap),
            A_Overlap(l_x1_p) / (A_p(u_x1_p) + A_gt - A_Overlap)] 

        As A_Overlap and A_Union are already defined as interval, the IoU calculation is an interval multiplication and division.

        
            
        """
        # vanilla IBP reluval
        A_union_reluval = self.union_reluval()
        A_union_reciprocal_reluval = A_union_reluval.reciprocal_positive()
        A_overlap_reluval = self.overlap_reluval()
        IoU_vanilla_reluval = A_overlap_reluval * A_union_reciprocal_reluval


        if display == True: 
            dict_iou_reluval = {"A_gt": self.hyperrect.area(),
                                "A_union": A_union_reluval.aslist(),
                                "1/A_union": A_union_reluval.reciprocal_positive().aslist(),
                                "A_overlap":A_overlap_reluval.aslist(), 
                                "A_p":self.hyperrect_interval.area().aslist() , 
                                "IoU":IoU_vanilla_reluval}
            for key, values in dict_iou_reluval.items(): 
                if key != "IoU":
                    print(key, values)
                else:
                    print(key, values.aslist())
        if IoU_vanilla_reluval.aslist()[1] > 1:
            return([IoU_vanilla_reluval.aslist()[0], 1])
        else: 
            return(IoU_vanilla_reluval.aslist())


        

    def iou(self, display = False):
        start_vanilla =  time.time()
        vanilla = self.iou_reluval()
        end_vanilla = time.time()

        start_extension = time.time()
        extension = self.iou_optim()
        end_extension = time.time() 


        dict_iou = { "IoU_vanilla":vanilla,
                     "tmps_vanilla": end_vanilla-start_vanilla,
                     "IoU_extension":extension,
                     "tmps_extension":end_extension-start_extension}

        if display == True: 
            for key, values in dict_iou.items(): 
                    print(key, values)
        else:
            return(dict_iou)

    def iou_display(self, display = False):
        start_vanilla =  time.time()
        vanilla = self.iou_reluval()
        end_vanilla = time.time()

        start_extension = time.time()
        extension = self.iou_optim()
        end_extension = time.time() 


        dict_iou = { "IoU_vanilla":vanilla,
                     "tmps_vanilla": end_vanilla-start_vanilla,
                     "IoU_extension":extension,
                     "tmps_extension":end_extension-start_extension, 
                     }

        if display == True: 
            for key, values in dict_iou.items(): 
                    print(key, values)
        else:
            return(dict_iou)

    
    def plot(self, optim_plot = True):
        """
        This function plot the different rectangles by using plot defined in class Hyperrectangle_interval and Hyperrectangle
        """

        if optim_plot == True: 
            fig = go.Figure(data = self.hyperrect_interval.plot(returnV=True).data 
                            + self.hyperrect.plot(returnV=True, name="ground_truth").data
                            + self.optim_rec().plot(returnV=True, name="optim").data)
        else: 
            fig = go.Figure(data = self.hyperrect_interval.plot(returnV=True).data 
                + self.hyperrect.plot(returnV=True, name="ground_truth").data)
        fig.show()
        

