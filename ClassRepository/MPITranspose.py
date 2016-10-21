# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class MPITranspose:
    
    import numpy as np
    from mpi4py import MPI    
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
    
    def DoMPITranspose(self,v,lx,lz_half,nproc,my_id):
        comm=self.MPI.COMM_WORLD
        isize=(lz_half+1)*lx*lx
        tmp1=self.np.zeros((lx,lx,lz_half+1),dtype='complex64')
        tmp2=self.np.zeros((lx,lx,lz_half+1),dtype='complex64')
    
        for i in range(1,nproc):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc

            j=nzp*lx
            j1=(nzp+1)*lx
            tmp1[:,0:lx,:] = v[:,j:j1,:]

            req1=comm.Isend([tmp1,self.MPI.COMPLEX],dest=nzp,  tag=i)
            req2=comm.Irecv([tmp2,self.MPI.COMPLEX],source=nzm,tag=i)
            self.MPI.Request.Waitall([req1,req2])

            js = nzp*lx
        
            for i in range(lx):
                for j in range(lx):
                    j1=js+j
                    v[i,j1,:] = tmp2[j,i,:]
                
        #... diagonal block transpose
        j  =my_id*lx
        j1 =(my_id+1)*lx
        tmp1[:,0:lx,:] = v[:,j:j1,:]

        js=my_id*lx
        for i in range(lx):
            for j in range(lx):
                j1=js+j
                v[i,j1,:] = tmp1[j,i,:]

        #... adjust the position of blocks which is not on diagonal line
        for i in range(1,nproc//2):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc
            j  =nzp*lx
            j1 =lx*(nzp+1)
            k  =nzm*lx
            k1 =lx*(nzm+1)
            tmp1[:,0:lx,:] = v[:,j:j1,:]
            v[:,j:j1,:] = v[:,k:k1,:]
            v[:,k:k1,:] = tmp1[:,0:lx,:]
        
    
        del tmp1
        del tmp2
        return