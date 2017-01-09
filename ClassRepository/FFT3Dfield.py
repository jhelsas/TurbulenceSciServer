# CREATED BY MOHAMMAD DANISH, JOHNS HOPKINS UNIVERSITY
# LAST MODIFIED: OCTOBER 05, 2016

# DESCRIPTION OF THE CLASS:
# This class converts 3D velocity (any) field from physical space to fourier space.

# PROVIDE EXAMPLE:

class FFT3Dfield:
    # space for importing namespace
    import pyfftw as ft 
    import numpy as np
    from mpi4py import MPI
    from MPITranspose import MPITranspose
        
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        self.myMPITranspose=self.MPITranspose()
        return
        
        
    def GetFFT3Dfield(self,v,lx,ly,lz,nproc,my_id):
        lz_half=lz//2
        
        temp_v=self.ft.zeros_aligned((lx,ly,lz),dtype='float32')
        cv=self.ft.zeros_aligned((lx,ly,lz_half+1),dtype='complex64')
        
        rfftzf=self.ft.FFTW(temp_v, cv,direction='FFTW_FORWARD',axes=(2,),flags=('FFTW_MEASURE',))
        fftyf =self.ft.FFTW(cv,cv,direction='FFTW_FORWARD',axes=(1,),flags=('FFTW_MEASURE',))

        self.mpifft3DRC(v,cv,rfftzf,fftyf,lx,ly,lz_half,nproc,my_id)
        
        return cv
    
    # 3D Real to Complex FFT
    def  mpifft3DRC(self,v,cv,rfftzf,fftyf,lx,ly,lz_half,nproc,my_id):
        rfftzf.update_arrays(v,cv)
        rfftzf.execute()
        fftyf.update_arrays(cv,cv)
        fftyf.execute() 
        self.myMPITranspose.DoMPITranspose(cv,lx,lz_half,nproc,my_id)
        fftyf.execute()

        alpha2=1.0/(float(ly)*float(ly)*float(ly))
        cv[:,:,:]=cv[:,:,:]*alpha2
        return
    
class FFT3Dfield_old:
    # space for importing namespace
    import pyfftw as ft 
    import numpy as np
    from mpi4py import MPI
    from MPITranspose import MPITranspose
        
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        self.myMPITranspose=self.MPITranspose()
        return
        
        
    def GetFFT3Dfield_old(self,v,lx,ly,lz,nproc,my_id):
        lz_half=lz//2
        
        cv=self.ft.zeros_aligned((lx,ly,lz_half+1),dtype='complex64')
        
        rfftzf=self.ft.FFTW(v, cv,direction='FFTW_FORWARD',axes=(2,),flags=('FFTW_MEASURE',))
        fftyf =self.ft.FFTW(cv,cv,direction='FFTW_FORWARD',axes=(1,),flags=('FFTW_MEASURE',))

        self.mpifft3DRC(v,cv,rfftzf,fftyf,lx,ly,lz_half,nproc,my_id)
        
        return cv
    
    # 3D Real to Complex FFT
    def  mpifft3DRC_old(self,v,cv,rfftzf,fftyf,lx,ly,lz_half,nproc,my_id):
        rfftzf.update_arrays(v,cv)
        rfftzf.execute()
        fftyf.update_arrays(cv,cv)
        fftyf.execute() 
        self.myMPITranspose.DoMPITranspose(cv,lx,lz_half,nproc,my_id)
        fftyf.execute()

        alpha2=1.0/(float(ly)*float(ly)*float(ly))
        cv[:,:,:]=cv[:,:,:]*alpha2
        return