import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def menger(x,y,z,lvl):
    lx = x
    ly = y
    lz = z
    for k in range(lvl):
        digx = math.floor(3*lx)
        digy = math.floor(3*ly)
        digz = math.floor(3*lz)
        if (digx==1)&(digy==1):
            return 0
        elif (digy==1)&(digz==1):
            return 0
        elif (digx==1)&(digz==1):
            return 0
        
        lx = 3.*lx-digx
        ly = 3.*ly-digy
        lz = 3.*lz-digz
        
    return 1    

vmenger = np.vectorize(menger)