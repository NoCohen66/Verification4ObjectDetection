import torch

import sys
sys.path.append('../')

class IntervalTensor:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        assert lower.shape == upper.shape
        self.lower = lower
        self.upper = upper
        self.shape = lower.shape
        self.device = lower.device
    
    def __repr__(self):
        return f'Lower:\n{self.lower}\nUpper:\n{self.upper}'
        
    def __getitem__(self, key):
        return IntervalTensor(self.lower[key], self.upper[key])
    
    def __neg__(self):
        return IntervalTensor(-self.upper, -self.lower)
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return IntervalTensor(self.lower + other.lower, self.upper + other.upper)
        elif isinstance(other, (int, float)):
            return IntervalTensor(self.lower + other, self.upper + other)
        elif isinstance(other, tuple):
            t = torch.tensor(other, device=self.device).unsqueeze(-1).unsqueeze(-1)
            return IntervalTensor(self.lower + t, self.upper + t)
        else:
            raise TypeError(f"Unsupported operand type(s) for +/-: '{self.__class__}' and '{type(other)}'")
    __radd__ = __add__
    
    def __sub__(self, other):
        return self + -other
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            # [a, b] * [c, d]
            a, b, c, d = self.lower, self.upper, other.lower, other.upper
            ac, ad, bc, bd = a*c, a*d, b*c, b*d
            mi = torch.minimum(torch.minimum(ac, ad), torch.minimum(bc, bd))
            ma = torch.maximum(torch.maximum(ac, ad), torch.maximum(bc, bd))
            return IntervalTensor(mi, ma)
        elif isinstance(other, (int, float)):
            if other < 0:
                return IntervalTensor(self.upper * other, self.lower * other)
            else:
                return IntervalTensor(self.lower * other, self.upper * other)
        elif isinstance(other, torch.Tensor):
            # this is to handle the case where we want to zero out some terms via a Boolean tensor
            return IntervalTensor(self.lower * other, self.upper * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            # [a, b] / [c, d]
            a, b, c, d = self.lower, self.upper, other.lower, other.upper
            ac, ad, bc, bd = a/c, a/d, b/c, b/d
            mi = torch.minimum(torch.minimum(ac, ad), torch.minimum(bc, bd))
            ma = torch.maximum(torch.maximum(ac, ad), torch.maximum(bc, bd))
            return IntervalTensor(mi, ma)
        elif isinstance(other, (int, float)):
            if other < 0:
                return IntervalTensor(self.upper / other, self.lower / other)
            else:
                return IntervalTensor(self.lower / other, self.upper / other)
        elif isinstance(other, tuple):
            # should only need this for dividing by stddev, so assume all divisors are positive here
            # assumes tuple of scalars
            t = torch.tensor(other, device=self.device).unsqueeze(-1).unsqueeze(-1)
            return IntervalTensor(self.lower / t, self.upper / t)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{self.__class__}' and '{type(other)}'")
    
    def flatten(self, start_dim=0, end_dim=-1):
        return IntervalTensor(self.lower.flatten(start_dim, end_dim), self.upper.flatten(start_dim, end_dim))
    
    def unsqueeze(self, dim):
        return IntervalTensor(self.lower.unsqueeze(dim), self.upper.unsqueeze(dim))

    def view(self, *shape):
        return IntervalTensor(self.lower.view(shape), self.upper.view(shape))
    
    def reshape(self, shape):
        return IntervalTensor(self.lower.reshape(shape), self.upper.reshape(shape))

# element-wise multiplication for interval tensors when ALL entries are positive
def mul_pos(a, b):
    return IntervalTensor(a.lower * b.lower, a.upper * b.upper)

def abs_i(a):
    overlaps = torch.logical_and(a.upper >= 0, a.lower <= 0)
    return -a*(a.upper < 0) + a*(a.lower > 0) + \
        IntervalTensor(torch.full(a.shape, 0, device=a.device), torch.maximum(a.upper, -a.lower))*overlaps

def max_i(a, b):
    if isinstance(a, (int, float)):
        return max_i(IntervalTensor(torch.full(b.shape, a, device=b.device), torch.full(b.shape, a, device=b.device)), b)
    elif isinstance(b, (int, float)):
        return max_i(IntervalTensor(torch.full(a.shape, b, device=a.device), torch.full(a.shape, b, device=a.device)), a)
    else:
        assert a.shape == b.shape
        overlaps = torch.logical_and(a.lower <= b.upper, a.upper >= b.lower)
        return a*(a.lower > b.upper) + b*(a.upper < b.lower) + \
            IntervalTensor(torch.maximum(a.lower, b.lower), torch.maximum(a.upper, b.upper))*overlaps

def relu_i(a):
    return max_i(0, a)

def min_i(a, b):
    if isinstance(a, (int, float)):
        return min_i(IntervalTensor(torch.full(b.shape, a, device=b.device), torch.full(b.shape, a, device=b.device)), b)
    elif isinstance(b, (int, float)):
        return min_i(IntervalTensor(torch.full(a.shape, b, device=a.device), torch.full(a.shape, b, device=a.device)), a)
    else:
        assert a.shape == b.shape
        overlaps = torch.logical_and(a.lower <= b.upper, a.upper >= b.lower)
        return a*(a.upper < b.lower) + b*(a.lower > b.upper) + \
            IntervalTensor(torch.minimum(a.lower, b.lower), torch.minimum(a.upper, b.upper))*overlaps

# assumes values in [-pi/2, pi/2]
def sin_interval(lower, upper):
    sin_l = torch.sin(lower)
    sin_u = torch.sin(upper)
    return sin_l, sin_u

# assumes values in [-pi/2, pi/2]
def cos_interval(lower, upper):
    if upper.item() > 0 and lower.item() < 0:
        cos_u = torch.tensor([1], device=upper.device)
        cos_l = torch.minimum(torch.cos(lower), torch.cos(upper))
    else:
        if upper.item() <= 0:
            cos_l = torch.cos(lower)
            cos_u = torch.cos(upper)
        else:
            cos_l = torch.cos(upper)
            cos_u = torch.cos(lower)
    return cos_l, cos_u

# assumes theta is in [-pi/2, pi/2]
def sin_i(theta):
    sin_l, sin_u = sin_interval(theta.lower, theta.upper)
    return IntervalTensor(sin_l, sin_u)
    
# assumes theta is in [-pi/2, pi/2]
def cos_i(theta):
    cos_l, cos_u = cos_interval(theta.lower, theta.upper)
    return IntervalTensor(cos_l, cos_u)
