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

libdir='/home/idies/workspace/persistent/Scripts/ClassRepository'
workdir=libdir+'/'

sys.path.insert(0, libdir)

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

## Populate velocity field from the Database
comm.Barrier(); t1=MPI.Wtime()
SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Load velocity field cost: {0:.2f} seconds\n'.format(t2-t1))

## Transform velocity field from Physical to Fourier space
comm.Barrier(); t1=MPI.Wtime()
myFFT3Dfield=FFT3Dfield()
cvx=myFFT3Dfield.GetFFT3Dfield(vx,lx,ly,lz,nproc,my_id)
cvy=myFFT3Dfield.GetFFT3Dfield(vy,lx,ly,lz,nproc,my_id)
cvz=myFFT3Dfield.GetFFT3Dfield(vz,lx,ly,lz,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('3 FFTs cost: {0:.2f} seconds\n'.format(t2-t1))

#cvx=ft.zeros_aligned((lx,ly,lz_half+1),dtype='complex64')
#cvy=ft.zeros_aligned((lx,ly,lz_half+1),dtype='complex64')
#cvz=ft.zeros_aligned((lx,ly,lz_half+1),dtype='complex64')

## Transform velocity field from Fourier to Physical space
comm.Barrier(); t1=MPI.Wtime()
myIFFT3Dfield=IFFT3Dfield()
vx=myIFFT3Dfield.GetIFFT3Dfield(cvx,lx,ly,lz,nproc,my_id)
vy=myIFFT3Dfield.GetIFFT3Dfield(cvy,lx,ly,lz,nproc,my_id)
vz=myIFFT3Dfield.GetIFFT3Dfield(cvz,lx,ly,lz,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('3 IFFTs cost: {0:.2f} seconds\n'.format(t2-t1))

## Get energy spectrum in Fourier space
comm.Barrier(); t1=MPI.Wtime()
myEnergySpc=EnergySpectrum()
k2,ek=myEnergySpc.GetSpectrum(vx,vy,vz,lx,ly,lz,nek,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Compute E(k) cost: {0:.2f} seconds\n'.format(t2-t1))
    sys.stdout=open(workdir+'ek.txt','wt')
    for i in range(nek): 
        sys.stdout.write('{0:15.1f} {1:15.6e}\n'.format(i+1,ek[i]))
    sys.stdout.close()
    sys.stdout=scrout