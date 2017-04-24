# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import pyfftw as ft 
import numpy as np
from mpi4py import MPI
from pyJHTDB import libJHTDB
import sys

class DataDownload:
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
    
    def DwnldNsaveVelDataOnVM(self,dirName,fileNameInitial,dataset_name,time,lx,ly,lz,nproc,my_id,auth_token):
        vx,vy,vz=DownldData4Velocity(dataset_name,time,lx,ly,lz,nproc,my_id,auth_token)
        self.SaveVelDataOnVM(dirName,fileNameInitial,vx,vy,vz,nproc,my_id)
        return
    
    def DownldData4Velocity(self,dataset_name,time,lx,ly,lz,nproc,my_id,auth_token):
        chkSz=32 #This is the maximum possible. May be increased in future depending on network bandwidth
        slabs=lx//chkSz
        lJHTDB=libJHTDB(auth_token)
        lJHTDB.initialize()#NOTE: datbase returns Velcocity as [lz,ly,lx,3]
        for k in range(slabs):
            start=np.array([my_id*lx+k*chkSz,0,0],dtype=np.int)
            width=np.array([chkSz,ly,lz],dtype=np.int)
            uAll=lJHTDB.getRawData(time,start,width,data_set=dataset_name,getFunction='Velocity')
            if(k==0):
                vx=uAll[:,:,:,0]
                vy=uAll[:,:,:,1]
                vz=uAll[:,:,:,2]
            else:
                vx=np.concatenate((vx,uAll[:,:,:,0]),axis=2) #axis=2=> the index of lx
                vy=np.concatenate((vy,uAll[:,:,:,1]),axis=2)
                vz=np.concatenate((vz,uAll[:,:,:,2]),axis=2)
        lJHTDB.finalize()
        u=ft.zeros_aligned((lx,ly,lz),dtype='float32')
        v=ft.zeros_aligned((lx,ly,lz),dtype='float32')
        w=ft.zeros_aligned((lx,ly,lz),dtype='float32')
        u[:,:,:]=np.transpose(vx)
        v[:,:,:]=np.transpose(vy)
        w[:,:,:]=np.transpose(vz)
        return u,v,w
    
    def SaveVelDataOnVM(self,dirName,fileNameInitial,vx,vy,vz,nproc,my_id):
        outfile=dirName+fileNameInitial+'_'+str(my_id)
        np.savez(outfile,vx=vx,vy=vy,vz=vz,nproc=nproc)
        return
    
    def LoadVelDataFromVM(self,dirName,fileNameInitial,nproc,my_id):
        outfile = dirName+fileNameInitial+'_'+str(my_id)+'.npz'
        myfiles = np.load(outfile)
        nprocVM=int(myfiles['nproc'])

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
            myfiles_temp = np.load(dirName+fileNameInitial+'_'+str(changed_id)+'.npz')            
            vx=myfiles_temp['vx']
            vy=myfiles_temp['vy']
            vz=myfiles_temp['vz']
            
            for ic in range(int(changed_id+1),int(changed_id+fact)):
                myfiles_temp = np.load(dirName+fileNameInitial+'_'+str(ic)+'.npz')
                vx_temp=np.append(vx,myfiles_temp['vx'],axis=0)
                vy_temp=np.append(vy,myfiles_temp['vy'],axis=0)
                vz_temp=np.append(vz,myfiles_temp['vz'],axis=0)
        
        if nproc>nprocVM: #Asking with more resources: Make sure that they are power of 2
            print('Data loading fails. At present it is not ready to handle more resources than actually downloaded!!')
            return
        
        return vx_temp,vy_temp,vz_temp
    
    def DownldData4Pressure(self,dataset_name,time,lx,ly,lz,my_id,auth_token):
        chkSz=32 #This is the maximum possible. May be increased in future depending on network bandwidth
        slabs=lx//chkSz
        lJHTDB=libJHTDB(auth_token)
        lJHTDB.initialize()#NOTE: datbase returns Pressure as [lz,ly,lx]
        for k in range(slabs):
            start=np.array([my_id*lx+k*chkSz,0,0],dtype=np.int)
            width=np.array([chkSz,ly,lz],dtype=np.int)
            pSlab=lJHTDB.getRawData(time,start,width,data_set=dataset_name,getFunction='Pressure')
            if(k==0):
                pAll=pSlab[:,:,:]
            else:
                pAll=np.concatenate((pAll,pSlab[:,:,:]),axis=2) #axis=2=> the index of lx
        lJHTDB.finalize()
        p=ft.zeros_aligned((lx,ly,lz),dtype='float32')
        p[:,:,:]=np.transpose(pAll)
        return p
    
    def SavePresDataOnVM(self,dirName,fileNameInitial,p,nproc,my_id):
        outfile=dirName+fileNameInitial+'_'+str(my_id)
        np.savez(outfile,p=p,nproc=nproc)
        return
    
    def LoadPresDataFromVM(self,dirName,fileNameInitial,nproc,my_id):
        outfile = dirName+fileNameInitial+'_'+str(my_id)+'.npz'
        myfiles = np.load(outfile)
        nprocVM=int(myfiles['nproc'])

        if nproc==nprocVM:
            p_temp=myfiles['p']

        if nproc<nprocVM: #Asking with less resources: Make sure that they are power of 2
            fact=nprocVM/nproc
            if fact%2 != 0:
                print('Oops! the entered number of processors are not a power of 2. Try again with nproc= 2^n')
            
            #Combine the data and send it to the user
            changed_id=int(my_id*fact)
            myfiles_temp = np.load(dirName+fileNameInitial+'_'+str(changed_id)+'.npz')            
            p=myfiles_temp['p']
            
            for ic in range(int(changed_id+1),int(changed_id+fact)):
                myfiles_temp = np.load(dirName+fileNameInitial+'_'+str(ic)+'.npz')
                p_temp=np.append(p,myfiles_temp['p'],axis=0)
        
        if nproc>nprocVM: #Asking with more resources: Make sure that they are power of 2
            print('Data loading fails. At present it is not ready to handle more resources than actually downloaded!!')
            return
        
        return p_temp
    
    def DownldData_cSOAP(self,dataset_name,time,lx,ly,lz,nproc,my_id,auth_token):
        vx=ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vy=ft.zeros_aligned((lx,ly,lz), dtype='float32')
        vz=ft.zeros_aligned((lx,ly,lz), dtype='float32')
        SOAPtdb.loadvel(vx,vy,vz,lx,ly,lz,my_id)
        return vx,vy,vz