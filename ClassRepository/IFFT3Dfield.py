# CREATED BY MOHAMMAD DANISH, JOHNS HOPKINS UNIVERSITY
# LAST MODIFIED: OCTOBER 05, 2016

# DESCRIPTION OF THE CLASS:
# This class converts 3D velocity (any) field from Fourier space to Physical space.

# PROVIDE EXAMPLE:

class IFFT3Dfield:
    # space for importing namespace
    import pyfftw as ft 
    import numpy as np
    from mpi4py import MPI
    from MPITranspose import MPITranspose
        
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        
        return
            
    # 3D Complex to Real FFT
    def GetIFFT3Dfield(self,cv,lx,ly,lz,nproc,my_id):
        lz_half=lz//2
        
        temp_cv=self.ft.zeros_aligned((lx,ly,lz_half+1), dtype='complex64')        
        v=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        
        fftyb =self.ft.FFTW(temp_cv,temp_cv,direction='FFTW_BACKWARD',axes=(1,),flags=('FFTW_MEASURE',))
        rfftzb=self.ft.FFTW(temp_cv,v,direction='FFTW_BACKWARD',axes=(2,),flags=('FFTW_MEASURE',))
        
        cvt=self.ft.zeros_aligned((lx,ly,lz_half+1), dtype='complex64')
        self.np.copyto(cvt,cv)
        
        fftyb.update_arrays(cvt,cvt)
        fftyb.execute()
        
        myMPITranspose=self.MPITranspose()
        myMPITranspose.DoMPITranspose(cvt,lx,lz_half,nproc,my_id)
        
        fftyb.execute()
        rfftzb.update_arrays(cvt,v)        
        rfftzb.execute()
        return v

class IFFT3Dfield_old:
    # space for importing namespace
    import pyfftw as ft 
    import numpy as np
    from mpi4py import MPI
    from MPITranspose import MPITranspose
        
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        self.myMPITranspose=self.MPITranspose()
        return
    
    def GetIFFT3Dfield_old(self,cv,v,cvt,lx,ly,lz,nproc,my_id):
        lz_half=lz//2
        
        #v=self.ft.zeros_aligned((lx,ly,lz), dtype='float32')
        
        fftyb =self.ft.FFTW(cv,cv,direction='FFTW_BACKWARD',axes=(1,),flags=('FFTW_MEASURE',))
        rfftzb=self.ft.FFTW(cv, v,direction='FFTW_BACKWARD',axes=(2,),flags=('FFTW_MEASURE',))
        
        #cvt=self.ft.zeros_aligned((lx,ly,lz_half+1), dtype='complex64')
        
        self.np.copyto(cvt,cv)
        self.mpifft3DCR_old(v,cvt,rfftzb,fftyb,lx,ly,lz_half,nproc,my_id)
        
        return v
    
    # 3D Complex to Real FFT
    def  mpifft3DCR_old(self,v,cv,rfftzb,fftyb,lx,ly,lz_half,nproc,my_id):

        fftyb.update_arrays(cv,cv)
        fftyb.execute()

        self.myMPITranspose.DoMPITranspose(cv,lx,lz_half,nproc,my_id)

        fftyb.execute()
        rfftzb.update_arrays(cv,v)
        rfftzb.execute()
        return