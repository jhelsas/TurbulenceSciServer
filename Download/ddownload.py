from mpi4py import MPI
import sys
import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
#3.
sys.path.insert(0,'ClassRepository')
from DataDownload import DataDownload
#4.
comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
nproc = comm.Get_size()

if(my_id==0):
    sys.stdout.write('Starting with processors:%d\n' %(int(nproc)))
    
#1.
nx=isotropic1024coarse['nx']
ny=isotropic1024coarse['ny']
nz=isotropic1024coarse['nz']
#2.
nx=nx//nproc
sys.stdout.write('MPI id={0:4d} nx={1:6d} ny={2:6d} nz={3:6d}\n'. \
    format(my_id,nx,ny,nz))

#1.
auth_token='edu.jhu.meneveau-hiSmxkae'
#2.
dataset_name='isotropic1024coarse'
time=0.0
#3.
myDDL=DataDownload()
comm.Barrier(); t1=MPI.Wtime()
vx,vy,vz=myDDL.DownldData4Velocity(dataset_name,time,nx,ny,nz,nproc,my_id,auth_token)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Downloading cost: %0.2f seconds; with processors:%d\n' %(t2-t1,int(nproc)))
    
#1.
dirName='/home/admin/dataSnapshot/'
#2.
fileNameInitial='veldata_t0p0_myID'
#3.
comm.Barrier(); t1=MPI.Wtime()
myDDL=DataDownload()
myDDL.SaveVelDataOnVM(dirName,fileNameInitial,vx,vy,vz,nproc,my_id)
comm.Barrier(); t2=MPI.Wtime()
if(my_id==0):
    sys.stdout.write('Writing cost: %0.2f seconds; with processors:%d\n' %(t2-t1,int(nproc)))