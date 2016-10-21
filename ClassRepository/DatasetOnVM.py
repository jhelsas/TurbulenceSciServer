# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class DatasetOnVM:
    import pyfftw as ft 
    import numpy as np
    from mpi4py import MPI
    import SOAPtdb
    import sys

    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
    
    def SaveDataOnVM(self,dirName,fileName,nx,ny,nz,nproc,my_id):
        lx=nx//nproc
        ly=ny
        lz=nz
        ## Initialize the velocity field having a size of (lx,ly,lz)
        vx=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vy=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vz=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')

        ## Populate velocity field from the Database
        self.SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
        outfile =dirName+fileName+str(my_id)
        self.np.savez(outfile,vx=vx,vy=vy,vz=vz,nproc=nproc)
        return
            
    def LoadDataFromVM(self,dirName,fileName,nproc):
        outfile =dirName+fileName
        myfiles = self.np.load(outfile)
        if int(myfiles['nproc']) != nproc:
            print('Oops!  Unmatched nproc.  Try again with nproc= '+str(int(myfiles['nproc'])))
            return
        ############
        
        return myfiles['vx'],myfiles['vy'],myfiles['vz']