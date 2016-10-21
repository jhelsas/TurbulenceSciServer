import pyfftw as ft 
import numpy as np
from mpi4py import MPI
import SOAPtdb
import sys

saveDir='/home/idies/workspace/persistent/dataSnapshot/'
fileName='testdata'

comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
nproc = comm.Get_size()

nx=64
ny=64
nz=64

lx=nx//nproc
ly=ny
lz=nz
print(nproc)
print(lx)
## Initialize the velocity field having a size of (lx,ly,lz)
vx=ft.zeros_aligned((lx,ly,lz), dtype='float32')
vy=ft.zeros_aligned((lx,ly,lz), dtype='float32')
vz=ft.zeros_aligned((lx,ly,lz), dtype='float32')

## Populate velocity field from the Database
#comm.Barrier(); t1=MPI.Wtime()
#SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
#comm.Barrier(); t2=MPI.Wtime()
#if(my_id==0):
#    sys.stdout.write('Load velocity field cost: {0:.2f} seconds\n'.format(t2-t1))

#comm.Barrier(); t1=MPI.Wtime()
outfile =saveDir+fileName+'_myID_'+str(my_id)
np.savez(outfile,vx=vx,vy=vy,vz=vz,nproc=nproc)
#comm.Barrier(); t2=MPI.Wtime()
#if(my_id==0):
#    sys.stdout.write('Save velocity field cost: {0:.2f} seconds\n'.format(t2-t1))
