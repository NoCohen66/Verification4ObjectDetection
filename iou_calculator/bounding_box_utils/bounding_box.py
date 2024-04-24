import plotly.graph_objects as go



class Hyperrectangle:
    """Class representing a hyperrectangle defined by four bounds on the x and y axes."""
    
    def __init__(self, x_bl, x_tr, y_bl, y_tr):
        """
        Constructor of the Hyperrectangle class.
        
        x_tr: (float) top right point projected on the x-axis
        x_bl: (float) bottom left point projected on the x_axis
        y_tr: (float) top right point projected on the y-axis
        y_bl: (float) bottom left point projected on the y_axis
        """
        
        if not x_tr >= x_bl:
            raise ValueError("On x: The upper bounds must be greater or equal to the lower bounds")
        elif not y_tr >= y_bl:
            raise ValueError("On y: The upper bounds must be greater or equal to the lower bounds")
        else:
            self.x_tr = x_tr
            self.x_bl = x_bl
            self.y_tr = y_tr
            self.y_bl = y_bl
  
            
    
    def print_(self):
        print(f"x_tr: {self.x_tr}, x_bl: {self.x_bl}, y_tr: {self.y_tr}, y_bl: {self.y_tr}")
            
    def dict(self):
        """
        output:
        (dict):
                key (string): name of the coordinates top-right and bottom-left on x-axis and y-axis
                value (float): value of the coordinates on x-axis and y-axis
        """
        return({"x_tr": self.x_tr, "x_bl": self.x_bl, "y_tr": self.y_tr, "y_bl": self.y_bl})
    
    def dict_init(self):
        return({"x_tr": [self.x_tr], "x_bl": [self.x_bl], "y_tr": [self.y_tr], "y_bl": [self.y_bl]})
    
            
            
    def area(self):
        """
        (float): Calculate the area of the rectangle 
        """
        
        width = self.x_tr - self.x_bl
        height = self.y_tr - self.y_bl
        return width * height
    
    
    def is_area_not_zero(self):
        """
        (bool): True if the area is not null
        """
        length = self.x_tr - self.x_bl
        width = self.y_tr - self.y_bl
        return length > 0.05 and width > 0.05
    
    def plot(self, returnV=False, xaxes="x", yaxes="y", name="hyperrectangle_init"):
        x = [self.x_bl, self.x_tr, self.x_tr, self.x_bl, self.x_bl]
        y = [self.y_tr, self.y_tr, self.y_bl, self.y_bl, self.y_tr]
        fig = go.Figure(
            go.Scatter(
                x=x, y=y,
                fill="toself", 
                name=name
            )
        )
        fig.update_xaxes(title_text = xaxes)
        fig.update_yaxes(title_text = yaxes)

        if returnV == True: 
            return(fig)
        else:
            fig.show()

    def is_inside(self, point):
        """
        point: (list of float values) first value is the projection of the point on the x-axis, second y-axis

        output: 
        (bool) True if the point is in the rectangle else False.
        """
        return(self.x_bl <= point[0] <= self.x_tr and self.y_bl <= point[1] <= self.y_tr)

    def four_points(self):
        """
        output:
        (dict):
                key (string): name of the corners of the rectangle
                value (list): list with the x-axis coordinates and y-axis coordinates
        """
        return({
            "bottom_right" : [self.x_tr, self.y_bl],
            "bottom_left"  : [self.x_bl, self.y_bl],
            "top_right"    : [self.x_tr, self.y_tr],
            "top_left"     : [self.x_bl, self.y_tr]})

    def overlap(self, hyperrect_2):
        """
        Calculate the area of overlap between the two hyperrectangles, by calculating width and height.
        input: 
        hyperrect_2: (Hyperrectangle) 

        output: 
        (float) area of overlap

        """
        
        # Check if provided hyperrectangle is instance of the Hyperrectangle class
        if not isinstance(hyperrect_2, Hyperrectangle):
            raise ValueError("Hyperrect_2 should be instance of the Hyperrectangle class.")

        # Points for hyperrect_1 (predicted) that fit with latex description
        x_1_p = self.x_bl
        x_2_p = self.x_tr
        y_1_p = self.y_bl
        y_2_p = self.y_tr

        # Points for hyperrect_2 (ground truth) that fit with latex description
        x_1_gt = hyperrect_2.x_bl
        x_2_gt = hyperrect_2.x_tr
        y_1_gt = hyperrect_2.y_bl
        y_2_gt = hyperrect_2.y_tr

        """
        Calculate the width overlap between the two hyperrectangles.
        w_overlap = max(0, min(x_2_p, x_2_gt) - max(x_1_p, x_1_gt))
        """
        w_overlap = max(0, min(x_2_p, x_2_gt) - max(x_1_p, x_1_gt))

        """
        Calculate the height overlap between the two hyperrectangles.
        h_overlap = max(0, min(y_2_p, y_2_gt) - max(y_1_p, y_1_gt))
        """
        h_overlap =  max(0, min(y_2_p, y_2_gt) - max(y_1_p, y_1_gt))

        """
        Calculate the area of overlap between the two hyperrectangles.
        
        Formula derived from:
        A_overlap = w_overlap * h_overlap
        """
        return w_overlap * h_overlap