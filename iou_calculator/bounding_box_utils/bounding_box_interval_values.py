from iou_calculator.bounding_box_utils.interval import Interval
from iou_calculator.bounding_box_utils.bounding_box import Hyperrectangle
import pandas as pd
import plotly.graph_objects as go


class Hyperrectangle_interval:
    """Class representing a hyperrectangle where each coordinates of the rectangle have an interval of variation."""
    
    def __init__(self, x_1, x_2, y_1, y_2):
        """
        Constructor of the Hyperrectangle_interval class.
        
        x_2: (Interval) upper bound of the interval on the x-axis can be seen as x_tr
        x_1: (Interval) lower bound of the interval on the x-axis can be seen as x_bl
        y_2: (Interval) upper bound of the interval on the y-axis can be seen as y_tr
        y_1: (Interval) lower bound of the interval on the y-axis can be seen as y_bl
        """
        
        # Ensure all are of Interval type
        if not all(isinstance(val, Interval) for val in [x_1, x_2, y_1, y_2]):
            raise ValueError("All inputs must be instances of the Interval class.")

        # Ensure the intervals do not overlap
        if x_2 <= x_1 or y_2 <= y_1:
            raise ValueError("The upper bound intervals must be greater than the lower bound intervals.")
        
        self.x_2 = x_2
        self.x_1 = x_1
        self.y_2 = y_2
        self.y_1 = y_1
    
    def print_(self):
        print(f"x_2: {self.x_2.aslist()}, x_1: {self.x_1.aslist()}, y_2: {self.y_2.aslist()}, y_1: {self.y_1.aslist()}")
            
    def dict(self):
        return {"x_tr": self.x_2, "x_bl": self.x_1, "y_tr": self.y_2, "y_bl": self.y_1}
    
    def dict_init(self):
        return {"x_2": [self.x_2], "x_1": [self.x_1], "y_2": [self.y_2], "y_1": [self.y_1]}
            
    def min_hyperrect(self):
        # TO DO min_area_hyperrect
        """
        This function calculates the minimum rectangle that the range given by each coordinates gives.
        😉 Tips: Looking at a visualisation graph makes it easier to understand.
        """
        return(Hyperrectangle(x_bl=self.x_1.u,
                              y_bl=self.y_1.u, 
                              x_tr=self.x_2.l, 
                              y_tr=self.y_2.l))
        
    def max_hyperrect(self):
        """
        This function calculates the minimum rectangle that the range given by each coordinates gives.
        😉 Tips: Looking at a visualisation graph makes it easier to understand.
        """
        return(Hyperrectangle(x_bl=self.x_1.l,
                              y_bl=self.y_1.l, 
                              x_tr=self.x_2.u, 
                              y_tr=self.y_2.u))

    def area(self):
        """
        Calculating the area of the predicted bounding box:
            A_p(x1_p, x2_p, y1_p, y2_p) = (x2_p - x1_p) * (y2_p - y1_p)
            
        If we focus solely on x1_p, we can express A_p as:
            A_p(x1_p) = -C3 * x1_p + C1' #
            
            where:
            1. C1' = x2_p * (y2_p - y1_p)
            2. C3 = y2_p - y1_p (Note: C3 is always non-negative) 
            
            The negative coefficient for x1_p indicates that as x1_p increases, A_p decreases.
            This is consistent, as the minimum area is determined when x1_p is at its upper bound.
        """
        return(Interval(self.min_hyperrect().area(), self.max_hyperrect().area()))

    def is_area_not_zero(self):
        return self.min_hyperrect().area() > 0 

    def variation_bl(self):
        """
        As with this rectangle, it is possible to define a delimited zone where the bottom-left point can move, 
        as it can take on a range of values for each coordinate.
        """
        return(Hyperrectangle(x_bl=self.x_1.l,
                              x_tr=self.x_1.u, 
                              y_bl=self.y_1.l, 
                              y_tr=self.y_1.u))
    def variation_tr(self):
        """
        As with this rectangle, it is possible to define a delimited zone where the top-right point can move, 
        as it can take on a range of values for each coordinate.
        """
        return(Hyperrectangle(x_bl=self.x_2.l,
                              x_tr=self.x_2.u, 
                              y_bl=self.y_2.l, 
                              y_tr=self.y_2.u)) 

    def rectangle_database_func(self):
        """
        This function put into a pandas dataframe, the function express beforehand.
        It contains the names of this rectangles (minimum area, 
                                                    maximum  area, variations bottom left, 
                                                    variations top right, 
                                                    variations bottom left), their definitions and their area.
        
        """
        
        hyp = self.max_hyperrect()
        rectangle_database = pd.DataFrame(hyp.dict_init())
        rectangle_database['name_given'] = 'maximum area'
        rectangle_database['area_value'] = self.max_hyperrect().area()
        rectangle_database.loc[len(rectangle_database)] = {**self.min_hyperrect().dict(), 'name_given':"minimum area", 'area_value':self.min_hyperrect().area()}
        rectangle_database.loc[len(rectangle_database)] = {**self.variation_bl().dict(), 'name_given':"variations bottom left", 'area_value':self.variation_bl().area()}
        rectangle_database.loc[len(rectangle_database)] = {**self.variation_tr().dict(), 'name_given':"variations top right", 'area_value':self.variation_tr().area()}
        return(rectangle_database)


    def plot(self, returnV=False, xaxes="x", yaxes="y", name="hyperrectangle_init"):
        rectangle_database = self.rectangle_database_func()
        #rectangle_database["plot"] = rectangle_database.apply(lambda x: Hyperrectangle(x_bl=x.x_bl,x_tr=x.x_tr, y_bl=x.y_bl, y_tr=x.y_tr).plot(returnV=True, name=f"{x.name}").data, axis=1)
        #return(rectangle_database)
        fig_rectangle_database = go.Figure(data =rectangle_database.apply(lambda x: Hyperrectangle(x_bl=x.x_bl,x_tr=x.x_tr, y_bl=x.y_bl, y_tr=x.y_tr).plot(returnV=True, name=f"{x.name_given} (area={x.area_value})").data, axis=1).sum())
        
        if returnV == True: 
            return(fig_rectangle_database)
        else:
            fig_rectangle_database.show()

    def is_inside_max_area(self, point):
        """
        point: (list of float values) first value is the projection of the point on the x-axis, second y-axis

        output: 
        (bool) True if the point is in the maximum rectangle else False.
        """
        max_hyp = self.max_hyperrect()
        return(max_hyp.is_inside(point))

    def is_inside_min_area(self, point):
        """
        point: (list of float values) first value is the projection of the point on the x-axis, second y-axis

        output: 
        (bool) True if the point is in the minimum rectangle else False.
        """
        min_hyp = self.min_hyperrect()
        return(min_hyp.is_inside(point))

    def is_inside_variation_bl(self, point):
        """
        point: (list of float values) first value is the projection of the point on the x-axis, second y-axis

        output: 
        (bool) True if the point is in the variation domain of bottom left point.
        """
        bl_var = self.variation_bl()
        return(bl_var.is_inside(point))

    def is_inside_variation_tr(self, point):
        """
        point: (list of float values) first value is the projection of the point on the x-axis, second y-axis

        output: 
        (bool) True if the point is in the minimum rectangle else False.
        """
        tr_var = self.variation_tr()
        return(tr_var.is_inside(point))
        