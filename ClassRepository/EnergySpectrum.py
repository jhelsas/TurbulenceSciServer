# DESCRIPTION OF THE CLASS:
# Adding one more description line
# PROVIDE EXAMPLE:

class EnergySpectrum:
    
    import numpy as np
    from mpi4py import MPI
    from FFT3Dfield import FFT3Dfield
        
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
        
    def GetSpectrumFromRealField(self,vx,vy,vz,k2,lx,ly,lz,nek,nproc,my_id):
        myFFT3Dfield=self.FFT3Dfield()
        cvx=myFFT3Dfield.GetFFT3Dfield(vx,lx,ly,lz,nproc,my_id)
        cvy=myFFT3Dfield.GetFFT3Dfield(vy,lx,ly,lz,nproc,my_id)
        cvz=myFFT3Dfield.GetFFT3Dfield(vz,lx,ly,lz,nproc,my_id)
        
        k2[0,0,0]=1e-6
        ek=self.cal_newspec(cvx,cvy,cvz,k2,nek)
        return ek
    
    def GetSpectrumFromComplexField(self,cvx,cvy,cvz,k2,lx,ly,lz,nek,nproc,my_id):
        k2[0,0,0]=1e-6
        ek=self.cal_newspec(cvx,cvy,cvz,k2,nek)
        return ek
            
    def FindWavenumber(self,lx,ly,lz,my_id):
        lz_half=lz//2
        
        kx=self.np.zeros((lx,ly,lz_half+1), dtype='float32')
        ky=self.np.zeros((lx,ly,lz_half+1), dtype='float32')
        kz=self.np.zeros((lx,ly,lz_half+1), dtype='float32')
        for i in range(lx):
            ky[i,:,:]=(i+ly//2+lx*my_id)%ly-ly//2
        for j in range(ly):
            kz[:,j,:]=(j+ly//2)%ly-ly//2;
        for k in range(lz_half+1):
            kx[:,:,k]=k

        return kx, ky, kz
    
    def cal_newspec(self,cvx,cvy,cvz,k2,nek):
        tmp=(cvx*cvx.conj()+cvy*cvy.conj()+cvz*cvz.conj()).real
        tmp[:,:,0]=0.5*tmp[:,:,0]

        ekbins=self.np.linspace(0.5,nek+0.5,nek+1)
        k2rt=self.np.sqrt(k2)
        ekloc,bins=self.np.histogram(k2rt,range=(0.5,nek+0.5),bins=ekbins,weights=tmp)
        del k2rt
        del tmp

        ekloc=self.np.float32(ekloc)
        eksum=self.np.zeros(nek,dtype='float32')
        comm = self.MPI.COMM_WORLD
        comm.Reduce([ekloc,self.MPI.REAL],[eksum,self.MPI.REAL],op=self.MPI.SUM)
        
        ek=self.np.zeros(nek,dtype='float32')
        self.np.copyto(ek,eksum)
        return ek
    
    def cal_spec(self,cvx,cvy,cvz,k2,nek):
        tmp=(cvx*cvx.conj()+cvy*cvy.conj()+cvz*cvz.conj()).real
        tmp[:,:,0]=0.5*tmp[:,:,0]
        ekloc=self.np.zeros(nek+1,dtype='float32')
        eksum=self.np.zeros(nek+1,dtype='float32')

        comm = self.MPI.COMM_WORLD
        ks=self.np.arange(1,nek+1)
        for i in range(1,nek+1):
            ekloc[i]=self.np.sum(tmp[self.np.floor(self.np.sqrt(k2)+0.5)==i])
        comm.Reduce([ekloc,self.MPI.REAL],[eksum,self.MPI.REAL],op=self.MPI.SUM)

        ek=self.np.zeros(nek,dtype='float32')
        self.np.copyto(ek,eksum)
        return ek