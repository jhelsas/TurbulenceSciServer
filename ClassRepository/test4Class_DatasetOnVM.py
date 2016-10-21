import pyfftw as ft 
import numpy as np
from mpi4py import MPI
import math
import sys
import SOAPtdb

### This scripts test the three classes written for FFT, IFFT and Energy Spectrum ###
################################################################################
from DatasetOnVM import DatasetOnVM
from FFT3Dfield import FFT3Dfield
from IFFT3Dfield import IFFT3Dfield
from EnergySpectrum import EnergySpectrum
from Filters import Filters
from RandomNumberGenerator import RandomNumberGenerator

comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
nproc = comm.Get_size()

# Domain info:
nx=1024
ny=1024
nz=1024

sys.stdout.write('MPI id={0:4d} nx={1:6d} ny={2:6d} nz={3:6d}\n'. \
    format(my_id,nx,ny,nz))

lx=nx//nproc
ly=ny
lz=nz
lz_half=lz//2
nek=int(math.sqrt(2.0)/3*nx)

# load dataset from local drive:
comm.Barrier(); t1=MPI.Wtime()
dirName='/home/idies/workspace/persistent/dataSnapshot/'
fileName='veldata_myID_'+str(my_id)+'.npz'
myDataset=DatasetOnVM()
vx,vy,vz=myDataset.LoadDataFromVM(dirName,fileName,nproc)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Load velocity field cost: {0:.2f} seconds\n'.format(t2-t1))
    
## Get wavenumber:
comm.Barrier(); t1=MPI.Wtime()
myEnergySpc=EnergySpectrum()
kx,ky,kz=myEnergySpc.FindWavenumber(lx,ly,lz,my_id)
k2=np.zeros((lx,ly,lz_half+1), dtype='float32')
np.copyto(k2,kx*kx+ky*ky+kz*kz)
k2[0,0,0]=1e-6
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Wavenumber calculation cost: {0:.2f} seconds\n'.format(t2-t1))


## Get velocity field in wavenumber space:
comm.Barrier(); t1=MPI.Wtime()
myFFT3Dfield=FFT3Dfield()
cvx=myFFT3Dfield.GetFFT3Dfield(vx,lx,ly,lz,nproc,my_id)
cvy=myFFT3Dfield.GetFFT3Dfield(vy,lx,ly,lz,nproc,my_id)
cvz=myFFT3Dfield.GetFFT3Dfield(vz,lx,ly,lz,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('3 R-C FFTs cost: {0:.2f} seconds\n'.format(t2-t1))

############################# unfiltered #############################    
## Get energy spectrum in Fourier space
comm.Barrier(); t1=MPI.Wtime()
ek_all=myEnergySpc.GetSpectrumFromComplexField(cvx,cvy,cvz,k2,lx,ly,lz,nek,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Energy spec calculation cost: {0:.2f} seconds\n'.format(t2-t1))

############################# kappa_cutoff=100 #############################
kappa_c=100.0

## Filter the velocity field using the GAUSSIAN filter
comm.Barrier(); t1=MPI.Wtime()
myFilter=Filters()
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'gaussian')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'gaussian')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'gaussian')
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Gaussian filtering calculation cost: {0:.2f} seconds\n'.format(t2-t1))
    
## Get energy spectrum in Fourier space
ek_gaussian=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)

del cvx1
del cvy1
del cvz1

## Filter the velocity field using the SHARP filter
comm.Barrier(); t1=MPI.Wtime()
myFilter=Filters()
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'sharp')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'sharp')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'sharp')
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Sharp filtering calculation cost: {0:.2f} seconds\n'.format(t2-t1))

## Get energy spectrum in Fourier space
ek_sharp=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)

del cvx1
del cvy1
del cvz1

## Filter the velocity field using the BOX filter
comm.Barrier(); t1=MPI.Wtime()
myFilter=Filters()
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'box')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'box')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'box')
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Box filtering calculation cost: {0:.2f} seconds\n'.format(t2-t1))

## Get energy spectrum in Fourier space
ek_box=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)

del cvx1
del cvy1
del cvz1