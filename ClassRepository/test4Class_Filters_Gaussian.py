import pyfftw as ft 
import numpy as np
from mpi4py import MPI
import math
import sys
import SOAPtdb

### This scripts test the three classes written for FFT, IFFT and Energy Spectrum ###
################################################################################

from FFT3Dfield import FFT3Dfield
from IFFT3Dfield import IFFT3Dfield
from EnergySpectrum import EnergySpectrum
from Filters import Filters
from RandomNumberGenerator import RandomNumberGenerator

comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
nproc = comm.Get_size()

nx=1024
ny=1024
nz=1024
idp=8

scrout=sys.stdout
sys.stdout.write('MPI id={0:4d} nx={1:6d} ny={2:6d} nz={3:6d}\n'. \
    format(my_id,nx,ny,nz))

lx=nx//nproc
ly=ny
lz=nz
lz_half=lz//2
nek=int(math.sqrt(2.0)/3*nx)

## Initialize the velocity field having a size of (lx,ly,lz)
vx=ft.zeros_aligned((lx,ly,lz), dtype='float32')
vy=ft.zeros_aligned((lx,ly,lz), dtype='float32')
vz=ft.zeros_aligned((lx,ly,lz), dtype='float32')

## Generate random field (Complex) between -1 and 1
#myRandNumber=RandomNumberGenerator()
#cvx=myRandNumber.GetRandNumber_complex(-1.0,1.0,lx,ly,lz_half+1)
#cvy=myRandNumber.GetRandNumber_complex(-1.0,1.0,lx,ly,lz_half+1)
#cvz=myRandNumber.GetRandNumber_complex(-1.0,1.0,lx,ly,lz_half+1)

## Populate velocity field from the Database
comm.Barrier(); t1=MPI.Wtime()
SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Load velocity field cost: {0:.2f} seconds\n'.format(t2-t1))
    
## Get wavenumber:
myEnergySpc=EnergySpectrum()
kx,ky,kz=myEnergySpc.FindWavenumber(lx,ly,lz,my_id)
k2=np.zeros((lx,ly,lz_half+1), dtype='float32')
np.copyto(k2,kx*kx+ky*ky+kz*kz)
k2[0,0,0]=1e-6

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

############################# kappa_cutoff=150 #############################
## Filter the velocity field:
kappa_c=150.0
myFilter=Filters()
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'gaussian')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'gaussian')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'gaussian')

## Get energy spectrum in Fourier space
comm.Barrier(); t1=MPI.Wtime()
ek_150=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
del cvx1
del cvy1
del cvz1

############################# kappa_cutoff=100 #############################
## Filter the velocity field:
myFilter=Filters()
kappa_c=100.0
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'gaussian')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'gaussian')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'gaussian')

## Get energy spectrum in Fourier space
comm.Barrier(); t1=MPI.Wtime()
ek_100=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
del cvx1
del cvy1
del cvz1

############################# kappa_cutoff=50 #############################
## Filter the velocity field:
myFilter=Filters()
kappa_c=50.0
cvx1=myFilter.FilterTheComplexField(cvx,k2,kappa_c,'gaussian')
cvy1=myFilter.FilterTheComplexField(cvy,k2,kappa_c,'gaussian')
cvz1=myFilter.FilterTheComplexField(cvz,k2,kappa_c,'gaussian')

## Get energy spectrum in Fourier space
comm.Barrier(); t1=MPI.Wtime()
ek_50=myEnergySpc.GetSpectrumFromComplexField(cvx1,cvy1,cvz1,k2,lx,ly,lz,nek,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Compute E(k) cost: {0:.2f} seconds\n'.format(t2-t1))

del cvx1
del cvy1
del cvz1

#if(my_id==0):
#    sys.stdout.write('Compute E(k) cost: {0:.2f} seconds\n'.format(t2-t1))
#    sys.stdout=open(workdir+'ek.txt','wt')
#    for i in range(nek): 
#        sys.stdout.write('{0:15.1f} {1:15.6e}\n'.format(i+1,ek[i]))
#    sys.stdout.close()
#    sys.stdout=scrout