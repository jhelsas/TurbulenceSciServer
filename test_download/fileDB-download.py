import os
import io
import sys
import time
import pymp
import h5py
import urllib 
import ctypes
import pymorton as pym
import requests
import numpy as np
import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
from pyJHTDB import libJHTDB

os.environ['TMPDIR']='/home/idies/workspace/scratch'

Nx = isotropic1024coarse['nx']; Ny = isotropic1024coarse['ny']; Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']; Ly = isotropic1024coarse['ly']; Lz = isotropic1024coarse['lz']

dataset = 'isotropic1024coarse'
getFunction='Velocity'
t = 0.0; nx=Nx; ny=Ny; nz=Nz
chkSz = 32; slabs = nx//chkSz

print(os.environ['TMPDIR'])

t1 = time.time()

shu = pymp.shared.array((Nx,Ny,Nz), dtype='float32')
shv = pymp.shared.array((Nx,Ny,Nz), dtype='float32')
shw = pymp.shared.array((Nx,Ny,Nz), dtype='float32')

t2 = time.time()
sys.stdout.write('Alocating shared memory arrays: {0:.2f} seconds\n'.format(t2-t1))

t1 = time.time()

threads = 8

zmin = pym.interleave3(0,0,0)
zmax = pym.interleave3(511,511,511)+1
bs = 8
blobsize = bs*bs*bs

with pymp.Parallel(threads) as p:
    for idx in p.range(0,threads):
        
        if idx==0:
            print("Downloading")
        
        virgo_url = 'http://dsp060/disk0'+str(idx+1)+'/filedb/turbulence/turbdb10'+str(idx+1)+'_0.bin'
        d=urllib.urlopen(virgo_url).read()
        n=int(len(d)/4)
        _arr=np.frombuffer(d,dtype=np.float32,count=n)
        d=None
                
        ######################################
        
        if idx==0:
            print("z-reordering")
            
        nb = 512
        uBlock = np.zeros((nb,nb,nb),dtype='float32')
        vBlock = np.zeros((nb,nb,nb),dtype='float32')
        wBlock = np.zeros((nb,nb,nb),dtype='float32')
        
        for zindex in range(zmin,zmax,blobsize):
            coord = pym.deinterleave3(zindex)
            c0 = coord[0]; c1 = coord[1]; c2 = coord[2]
            
            blob = _arr[3*zindex:3*(zindex+blobsize)]
            block = np.reshape(blob,(bs,bs,bs,3),order='C')
            
            uBlock[c0:(c0+bs),c1:(c1+bs),c2:(c2+bs)] = np.transpose(block[:,:,:,0])
            vBlock[c0:(c0+bs),c1:(c1+bs),c2:(c2+bs)] = np.transpose(block[:,:,:,1])
            wBlock[c0:(c0+bs),c1:(c1+bs),c2:(c2+bs)] = np.transpose(block[:,:,:,2])
            
        #######################################
        
        if idx==0:
            print(uBlock[0,0,0])
            print(vBlock[0,0,0])
            print(wBlock[0,0,0])
                
        if idx==0:
            print("writing the blocks on the right places")
                
        Coord = pym.deinterleave3(idx)
        C0 = Coord[0]; C1 = Coord[1]; C2 = Coord[2]
        
        shu[C0*nb:(C0+1)*nb,C1*nb:(C1+1)*nb,C2*nb:(C2+1)*nb] = uBlock[:,:,:]
        shv[C0*nb:(C0+1)*nb,C1*nb:(C1+1)*nb,C2*nb:(C2+1)*nb] = vBlock[:,:,:]
        shw[C0*nb:(C0+1)*nb,C1*nb:(C1+1)*nb,C2*nb:(C2+1)*nb] = wBlock[:,:,:]
        
        ########################################
            
        del uBlock,vBlock,wBlock
        
t2 = time.time()
sys.stdout.write('Getting the data: {0:.2f} seconds\n'.format(t2-t1))

print(shu[shu==0])
print(shv[shv==0])
print(shw[shw==0])