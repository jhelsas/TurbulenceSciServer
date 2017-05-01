import os
import sys
import math
import numpy as np
from mpi4py import MPI

import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
from pyJHTDB import libJHTDB

Nx = isotropic1024coarse['nx']; Ny = isotropic1024coarse['ny']; Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']; Ly = isotropic1024coarse['ly']; Lz = isotropic1024coarse['lz']
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
if(rank==0):
    print("n_proc = "+str(nproc))
    print("rank = "+str(rank))

# Computational Domain

nx=Nx//nproc; ny=Ny; nz=Nz
nz_half=nz//2
nek=int(math.sqrt(2.0)/3*Nx)
time = 0.0

chkSz = 32
slabs = nx//chkSz

folder = "/home/jhelsas/scratch"
filename = "ref-enstrophy-"+str(rank)+".npz"
file = folder + "/" + filename

comm.Barrier(); t1=MPI.Wtime()
content = np.load(file)

w2 = np.zeros((nx,ny,nz), dtype='float32')
  
w2[:,:,:] = content['w2']
    
comm.Barrier(); t2=MPI.Wtime()

if(rank==0):
    print("Finished loading")
    sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))
        
folder = "/home/jhelsas/scratch"
filename = "ref-strainrate-"+str(rank)+".npz"
file = folder + "/" + filename

comm.Barrier(); t1=MPI.Wtime()
content = np.load(file)

S2 = np.zeros((nx,ny,nz), dtype='float32')
    
S2[:,:,:] = content['S2']
    
comm.Barrier(); t2=MPI.Wtime()

if(rank==0):
    print("Finished loading")
    sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))
    
w2 = 0.5*w2

avgO = np.average(w2)
avgOGl=np.zeros(1,dtype='float32')

comm.Allreduce([avgO,MPI.REAL],[avgOGl,MPI.REAL],op=MPI.SUM)
avgO = avgOGl[0]/nproc

########

avgE = np.average(S2)
avgEGl=np.zeros(1,dtype='float32')

comm.Allreduce([avgE,MPI.REAL],[avgEGl,MPI.REAL],op=MPI.SUM)
avgE = avgEGl[0]/nproc

########
    
avg = avgE

##########################

minw2 = w2.min()
maxw2 = w2.max()

minwGl=np.zeros(nproc,dtype='float32')
maxwGl=np.zeros(nproc,dtype='float32')

comm.Allgather([minw2,MPI.REAL],[minwGl,MPI.REAL])
comm.Allgather([maxw2,MPI.REAL],[maxwGl,MPI.REAL])

minO = minwGl.min()
maxO = maxwGl.max()

comm.Barrier()

##########################

minS2 = S2.min()
maxS2 = S2.max()

minS2Gl=np.zeros(nproc,dtype='float32')
maxS2Gl=np.zeros(nproc,dtype='float32')

comm.Allgather([minS2,MPI.REAL],[minS2Gl,MPI.REAL])
comm.Allgather([maxS2,MPI.REAL],[maxS2Gl,MPI.REAL])

minE = minS2Gl.min()
maxE = maxS2Gl.max()

comm.Barrier()

minJ = min(minO,minE)
maxJ = max(maxO,maxE)


if rank == 0:
    print(avgO,avgE,(avgE-avgO)/avgO)
    
if rank == 0:
    print("Separate : ",minO/avg,maxO/avg,minE/avg,maxE/avg)
    print("Joint : ",minJ/avg,maxJ/avg)

comm.Barrier()

if rank==0:
    print("log: ",np.log(minJ/avg),np.log(maxJ/avg))
    print("log_10: ",np.log(minJ/avg)/np.log(10),np.log(maxJ/avg)/np.log(10))
    
eta = 0.00280
x0 = y0 = z0 = 0.
dx = isotropic1024coarse['dx']
ner = int(1024*np.sqrt(3))
rbins = np.linspace(-0.5*dx,2*np.pi*np.sqrt(3)+0.5*dx,ner+1)

comm.Barrier(); t1=MPI.Wtime()

X = np.zeros((nx,ny,nz), dtype='float32')
Y = np.zeros((nx,ny,nz), dtype='float32')
Z = np.zeros((nx,ny,nz), dtype='float32')
r2 = np.zeros((nx,ny,nz), dtype='float32')

chi = np.zeros((nx,ny,nz), dtype='float32')

comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    sys.stdout.write('Alocating vectors: {0:.2f} seconds\n'.format(t2-t1))
    
comm.Barrier(); t1=MPI.Wtime()
for i in range(nx):
    X[i,:,:] = (i+rank*nx)*isotropic1024coarse['dx']
    
for j in range(ny):
    Y[:,j,:] = j*isotropic1024coarse['dy']
    
for k in range(nz):
    Z[:,:,k] = k*isotropic1024coarse['dz']
    
r2[:,:,:] = X[:,:,:]**2+Y[:,:,:]**2+Z[:,:,:]**2

r2rt = np.sqrt(r2)
comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    sys.stdout.write('Preparing the real domain for radial integration: {0:.2f} seconds\n'.format(t2-t1))
    
thresholds = [1]#[1,2,3,4,5,6,7,10,15,20,30,50]

tboxes = []

for t in thresholds:
    comm.Barrier(); t1=MPI.Wtime()
    
    Xs = X[w2 > t*avg]
    Ys = Y[w2 > t*avg]
    Zs = Z[w2 > t*avg]
    
    print(Xs.shape)
    
    hist = np.zeros((Xs.shape[0],3))
    
    hist[:,0] = Xs[:]
    hist[:,1] = Ys[:]
    hist[:,2] = Zs[:]
    
    count = []
    scales = np.logspace(np.log(2*425*eta),np.log(0.1*42.5*eta), num=250, endpoint=True, base=np.e)
    
    for L in scales:
        x1 = x0+isotropic1024coarse['lx']
        y1 = y0+isotropic1024coarse['ly']
        z1 = z0+isotropic1024coarse['lz']
        
        nx = int((x1-x0)/L)+1
        ny = int((y1-y0)/L)+1
        nz = int((z1-z0)/L)+1
        
        x1 = x0 + nx*L
        y1 = y0 + ny*L 
        z1 = z0 + nz*L
        
        H, edges = np.histogramdd(hist, bins=(nx,ny,nz), range=((x0,x1),(y0,y1),(z0,z1)), normed=True)
        
        Hglobal = np.zeros(H.shape,dtype='float64')
        comm.Allreduce([H,MPI.DOUBLE],[Hglobal,MPI.DOUBLE],op=MPI.SUM)
        
        Hn = Hglobal[:,:,:]
        Hn[Hn>0] = 1
        numBox = np.sum(Hn)
        gbox = np.zeros(1,dtype='float32')
        gbox[0] = numBox
        gbox = gbox[0]
        
        count.append(gbox)
    
    acount = np.array(count)
    
    tboxes.append(acount[:])
        
    comm.Barrier(); t2=MPI.Wtime()
    if(rank==0):
        sys.stdout.write('Computing boxcounting numbers: {0:.2f} seconds\n'.format(t2-t1))
        
if rank==0:
    tfboxes = np.array(tboxes)
    np.savez('boxcount-excursion-enstrophy-dims.npz',tfboxes = tfboxes)