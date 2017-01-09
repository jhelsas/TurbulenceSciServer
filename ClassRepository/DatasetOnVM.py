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
    
    def SaveDataOnVM(self,dirName,fileNameInitial,lx,ly,lz,nproc,my_id):
        #lx=nx//nproc
        #ly=ny
        #lz=nz
        ## Initialize the velocity field having a size of (lx,ly,lz)
        vx=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vy=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vz=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')

        ## Populate velocity field from the Database
        self.SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
        outfile =dirName+fileNameInitial+'_'+str(my_id)
        self.np.savez(outfile,vx=vx,vy=vy,vz=vz,nproc=nproc)
        return
            
    def LoadDataFromVM_old(self,dirName,fileNameInitial,nproc):
        outfile =dirName+fileNameInitial
        myfiles = self.np.load(outfile)
        if int(myfiles['nproc']) != nproc:
            print('Oops!  Unmatched nproc.  Try again with nproc= '+str(int(myfiles['nproc'])))
            return
        ############
        return myfiles['vx'],myfiles['vy'],myfiles['vz']
    
    def LoadDataFromVM(self,dirName,fileNameInitial,nproc,my_id,lx,ly,lz):
        outfile = dirName+fileNameInitial+'_'+str(my_id)+'.npz'
        myfiles = self.np.load(outfile)
        nprocVM=int(myfiles['nproc'])
        
        #vx_temp=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        #vy_temp=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        #vz_temp=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')

        if nproc==nprocVM:
            vx_temp=myfiles['vx']
            vy_temp=myfiles['vy']
            vz_temp=myfiles['vz']            

        if nproc<nprocVM: #Asking with less resources: Make sure that they are power of 2
            fact=nprocVM/nproc
            if fact%2 != 0:
                print('Oops! the entered number of processors are not a power of 2. Try again with nproc= 2^n')
            
            #Combine the data and send it to the user
            changed_id=int(my_id*fact)
            myfiles_temp = self.np.load(dirName+fileNameInitial+'_'+str(changed_id)+'.npz')            
            vx=myfiles_temp['vx']
            vy=myfiles_temp['vy']
            vz=myfiles_temp['vz']
            
            for ic in range(int(changed_id+1),int(changed_id+fact)):
                myfiles_temp = self.np.load(dirName+fileNameInitial+'_'+str(ic)+'.npz')
                vx_temp=self.np.append(vx,myfiles_temp['vx'],axis=0)
                vy_temp=self.np.append(vy,myfiles_temp['vy'],axis=0)
                vz_temp=self.np.append(vz,myfiles_temp['vz'],axis=0)

        return vx_temp,vy_temp,vz_temp