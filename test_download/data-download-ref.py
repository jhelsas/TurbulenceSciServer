import os
import sys
import math
import time
import numpy as np
import pyfftw as ft 
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
from pyJHTDB import libJHTDB

Nx = isotropic1024coarse['nx']
Ny = isotropic1024coarse['ny']
Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']
Ly = isotropic1024coarse['ly']
Lz = isotropic1024coarse['lz']

nproc = 1
rank = 0
nx=Nx//nproc
ny=Ny
nz=Nz
nz_half=nz//2
nek=int(math.sqrt(2.0)/3*Nx)
t = 0.0

chkSz = 32
slabs = nx//chkSz

begin = time.time()

from DataDownload import DataDownload

auth_token = "com.gmail.jhelsas-b854269a"

folder = "/home/idies/workspace/scratch"
filename = "ref-isotropic1024coarse-"+str(rank)+"-(t="+str(time)+")"+".npz"
filet = folder + "/" + filename

ddwnld = DataDownload()
vx,vy,vz = ddwnld.DownldData_pyJHTDB('isotropic1024coarse',t,nx,ny,nz,nproc,rank,auth_token)

#np.savez(filet,vx=vx,vy=vy,vz=vz,nproc=nproc)

end = time.time()