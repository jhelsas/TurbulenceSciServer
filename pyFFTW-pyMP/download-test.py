import os
import sys
import math
import time
import numpy as np
import pyfftw as ft 
from mpi4py import MPI
import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
from pyJHTDB import libJHTDB

from fft3d import FFT3Dfield_new
from EnergySpectrum import EnergySpectrum

Nx = isotropic1024coarse['nx']
Ny = isotropic1024coarse['ny']
Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']
Ly = isotropic1024coarse['ly']
Lz = isotropic1024coarse['lz']

from DataDownload import DataDownload

start = time.clock()

t = 0.0
auth_token = "com.gmail.jhelsas-b854269a"

folder = "/home/admin/scratch"
filename = "aws-isotropic1024coarse-(t="+str(time)+")"+".npz"
filename = folder + "/" + filename

nproc = 1
rank = 0
ddwnld = DataDownload()
vx,vy,vz = ddwnld.DownldData_pyJHTDB('isotropic1024coarse',t,Nx,Ny,Nz,nproc,rank,auth_token)

np.savez(filename,vx=vx,vy=vy,vz=vz)
    
end = time.clock()
print("time:",end-start)