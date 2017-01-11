# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import numpy as np
from mpi4py import MPI
import math
import numpy.linalg as la # Use eigh for symmetric matrix as it arrange eigenval in increasing order
    
class TensorTransformation:
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return

    # INPUT: Components of vectors A (=[a1,a2,a3]) and B (=[b1,b2,b3])
    # OUTPUT: Cos(\theta)=(A.B)/(|A|x|B|)
    def CosVectorVector(self,a1,a2,a3,b1,b2,b3):
        if (len(a1.shape)*len(a2.shape)*len(a3.shape)*len(b1.shape)*len(b2.shape)*len(b3.shape)>1.0):
            print('Change the input arrays of Cos2VectorVector function to 1D!!')
            return
        a_mag=np.sqrt(a1*a1+a2*a2+a3*a3)
        b_mag=np.sqrt(b1*b1+b2*b2+b3*b3)
        aDotb=a1*b1+a2*b2+a3*b3
        return abs(aDotb/(a_mag*b_mag))

    # VALID ONLY FOR SYMMETRIC TENSOR
    # INPUTS: Components of vector V (=[v1,v2,v3]) and 6 components of SYMMETRIC tensor S:
    #        _           _
    #       |s11  s12  s13|
    #     S=|s12  s22  s23|
    #       |s13  s23  s33|
    #        -           -
    # OUPUTS: |cos(V,alfa)|,|cos(V,beta)|,|cos(V,gama)|
    # NOTE: The function 'eigh' arranges the eigenvectors in the ascending order of eigenvalues [\gama<\beta<\alfa]
    # => v[:][0]=e_\gama; v[:][1]=e_\beta; v[:][2]=e_\alfa
    def CosVectorTensor(self,v1,v2,v3,s11,s22,s33,s12,s13,s23):
        if ((len(v1.shape)*len(v2.shape)*len(v3.shape)*len(s11.shape)*len(s22.shape)*\
             len(s33.shape)*len(s12.shape)*len(s13.shape)*len(s23.shape))>1.0):
            print('Change the input arrays of Cos2VectorTensor function to 1D!!')
            return
        
        lenS=len(s11)
        v_mag=np.sqrt(v1*v1+v2*v2+v3*v3)
        v1=v1/v_mag;v2=v2/v_mag;v3=v3/v_mag
        V=np.zeros((lenS,3,1))
        V[:,0,0]=v1;V[:,1,0]=v2;V[:,2,0]=v3
        
        S=np.zeros((lenS,3,3))
        S[:,0,0]=s11;S[:,0,1]=s12;S[:,0,2]=s13
        S[:,1,0]=s12;S[:,1,1]=s22;S[:,1,2]=s23
        S[:,2,0]=s13;S[:,2,1]=s23;S[:,2,2]=s33
        
        _,e=la.eigh(S)
        T=np.matmul(la.inv(e),V) # Transforms vector 'V' in the coordinate system 'e'
        # Return: |cos(V,alfa)|,|cos(V,beta)|,|cos(V,gama)|
        return abs(T[:,2,0]),abs(T[:,1,0]),abs(T[:,0,0])
    
    # VALID ONLY FOR SYMMETRIC TENSORS
    # INPUTS: Six components of SYMMETRIC tensors S and P
    #        _           _       _           _
    #       |s11  s12  s13|     |p11  p12  p13|
    #     S=|s12  s22  s23|   P=|p12  p22  p23|
    #       |s13  s23  s33|     |p13  p23  p33|
    #        --         --       --         --
    # OUPUTS: |cos(alfa,alfa)|,|cos(beta,beta)|,|cos(gama,gama)|,|cos(alfa,beta)|,|cos(alfa,gama)|,|cos(beta,gama)|
    # NOTE: The function 'eigh' arranges the eigenvectors in the ascending order of eigenvalues [\gama<\beta<\alfa]
    # => v[:][0]=e_\gama; v[:][1]=e_\beta; v[:][2]=e_\alfa
    def CosTensorTensor(self,s11,s22,s33,s12,s13,s23,p11,p22,p33,p12,p13,p23):# Not validated yet!!
        #TODO T=self.np.matmul(self.np.matmul(self.la.inv(Sv),P),Sv)
        if ((len(s11.shape)*len(s22.shape)*len(s33.shape)*len(s12.shape)*len(s13.shape)*len(s23.shape)*\
            len(p11.shape)*len(p22.shape)*len(p33.shape)*len(p12.shape)*len(p13.shape)*len(p23.shape))>1.0):
            print('Change the input arrays of Cos2TensorTensor function to 1D!!')
            return
        lenS=len(s11)
        S=np.zeros((lenS,3,3))
        S[:,0,0]=s11;S[:,0,1]=s12;S[:,0,2]=s13
        S[:,1,0]=s12;S[:,1,1]=s22;S[:,1,2]=s23
        S[:,2,0]=s13;S[:,2,1]=s23;S[:,2,2]=s33
        
        P=np.zeros((lenS,3,3))
        P[:,0,0]=p11;P[:,0,1]=p12;P[:,0,2]=p13
        P[:,1,0]=p12;P[:,1,1]=p22;P[:,1,2]=p23
        P[:,2,0]=p13;P[:,2,1]=p23;P[:,2,2]=p33
        
        _,es=la.eigh(S) # eigenvectors of S in ascending order of their eigenvalues [\gamma<beta<\alpha]
        _,ep=la.eigh(P) # eigenvectors of P in ascending order of their eigenvalues [\gamma<\beta<\alpha]
        T=np.matmul(la.inv(es),ep) # Transformation matrix
        e11=abs(T[:,0,0]);e22=abs(T[:,1,1]);e33=abs(T[:,2,2])
        e12=abs(T[:,0,1]);e13=abs(T[:,0,2]);e23=abs(T[:,1,2])
        # Return: |cos(alfa,alfa)|,|cos(beta,beta)|,|cos(gama,gama)|,|cos(alfa,beta)|,|cos(alfa,gama)|,|cos(beta,gama)|
        return e33,e22,e11,e23,e13,e12
    
    ####################################################################
    
    def CosAbsStrnrateVort_wrking(self,a11,a12,a13,a21,a22,a23,a31,a32,a33):
        a11=a11.flatten();a12=a12.flatten();a13=a13.flatten()
        a21=a21.flatten();a22=a22.flatten();a23=a23.flatten()
        a31=a31.flatten();a32=a32.flatten();a33=a33.flatten()
        lenA=len(a11)
        S=np.zeros((lenA,3,3))
        S[:,0,0]=a11;S[:,0,1]=0.5*(a12+a21);S[:,0,2]=0.5*(a13+a31)
        S[:,1,0]=S[:,0,1];S[:,1,1]=a22;S[:,1,2]=0.5*(a23+a32)
        S[:,2,0]=S[:,0,2];S[:,2,1]=S[:,1,2];S[:,2,2]=a33
        
        W=np.zeros((lenA,3,3))
        W[:,0,1]=0.5*(a12-a21);W[:,0,2]=0.5*(a13-a31)
        W[:,1,0]=-W[:,0,1];W[:,1,2]=0.5*(a23-a32)
        W[:,2,0]=-W[:,0,2];W[:,2,1]=-W[:,1,2]
                
        _,e=la.eigh(S) # e:eigenvectors of S in ascending order of eigenvalues [\gamma,\beta,\alpha]
        W_oriented=np.matmul(np.matmul(la.inv(e),W),e)
        vort1=2.*W_oriented[:,2,1];vort2=2.*W_oriented[:,0,2];vort3=2.*W_oriented[:,1,0]
        vort_mag=np.sqrt(vort1*vort1+vort2*vort2+vort3*vort3)
        return abs(vort3)/vort_mag,abs(vort2)/vort_mag,abs(vort1)/vort_mag # Descending order \alpha>\beta>\gamma