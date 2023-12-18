class Interval:
    def __init__(self, l,u=None):

        if u != None:
            if l > u:
                raise ValueError("Invalid interval bounds")
            self.l = l
            self.u = u
        else: 
            self.l = l
            self.u = l
            
    def __add__(self, other):
        return Interval(self.l + other.l, self.u + other.u)

    def __sub__(self, other):
        return Interval(self.l - other.u, self.u - other.l)

    def __mul__(self, other):
        # TO DO : verif all are positives
        min_val = min(self.l * other.l, self.l * other.u, self.u * other.l, self.u * other.u)
        max_val = max(self.l * other.l, self.l * other.u, self.u * other.l, self.u * other.u)
        return Interval(min_val, max_val)

    def display(self):
        print(f"[{self.l}, {self.u}]")

    def aslist(self):
        return([self.l, self.u])

    def reciprocal_positive(self):
        if self.l <= 0:
            raise ValueError("Reciprocal not defined for negatives or equal to 0 intervals.")
        return Interval(1 / self.u, 1 / self.l)
        
    def __lt__(self, other):
        return self.u < other.l

    def __le__(self, other):
        return self.u <= other.l

    def __gt__(self, other):
        return self.l > other.u

    def __ge__(self, other):
        return self.l >= other.u

    def one_value(self, returnV=False):
        if self.l == self.u:
            if returnV == True: 
                return(self.l)
            else:
                return(True)
        return(False)

    def contains(self, value):
        return self.u >= value >= self.l

    def max_reluval(self, value):
        if type(value) == Interval:
            raise ValueError("Value should not be an interval but a single data point.")    
        if self.u <= value:
            return(Interval(value, value))
        elif self.l >= value:
            return(Interval(self.l, self.u))
        else: 
            return(Interval(value, self.u))

    def min_reluval(self, value):
        if type(value) == Interval:
            raise ValueError("Value should not be an interval but a single data point.")    
        if self.l >= value:
            return(Interval(value, value))
        elif self.u <= value:
            return(Interval(self.l, self.u))
        else: 
            return(Interval(self.l, value))
        



