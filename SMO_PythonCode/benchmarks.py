
import numpy
import math

# define the function blocks 
def F1(x):
    s=numpy.sum(x**2);
    return s

def F_1(x, cent):
    sm =0 
    for t in numpy.nditer(x):
        sm += cent[str(int(t))]
    return math.fabs(sm) 

# define the function parameters 
def getFunctionDetails(a):
    
    # [name, lb, ub, dim, acc_err, obj_val]
    param = {	0: ["F1",-100,100,30,1.0e-5,0],
                        1:["F_1", 1, 379, 4, 1.0e-5, 1000],
            }
    return param.get(a, "nothing")



