
# coding: utf-8

# This notebook is to exemplify the downloading from the database and cache into disk. 

# In[2]:

#!ipcluster start -n 8 --engines=MPI --profile='mpi' # for parallel run: start the engines using terminal
from ipyparallel import Client
from IPython import get_ipython
rc = Client(profile='mpi')
ipython_shell = get_ipython()

# We first load the libraries necessary. We need to use pyfftw to define zeros_aligned arrays, and pyJHTDB for downloading. 

# In[5]:

ipython_shell.run_cell_magic('px', '', '# Import the libraries\n\nimport os\nimport sys\nimport math\nimport numpy as np\nimport pyfftw as ft \nfrom mpi4py import MPI\nimport pyJHTDB\nfrom pyJHTDB.dbinfo import isotropic1024coarse\nfrom pyJHTDB import libJHTDB')


# Then we need to initialize the associated variables, including domain, pyJHTDB and MPI variables. 

# In[6]:

ipython_shell.run_cell_magic('px', '', '\ncomm = MPI.COMM_WORLD; rank = comm.Get_rank(); nproc = comm.Get_size()\nif(rank==0):\n    print("n_proc = "+str(nproc))\n    print("rank = "+str(rank))\n\n# pyJHTDB parameters\n    \nNx = isotropic1024coarse[\'nx\']; Ny = isotropic1024coarse[\'ny\']; Nz = isotropic1024coarse[\'nz\']\nLx = isotropic1024coarse[\'lx\']; Ly = isotropic1024coarse[\'ly\']; Lz = isotropic1024coarse[\'lz\']\nauth_key = "com.gmail.jhelsas-b854269a"\n \n# Computational Domain\n\nnx=Nx//nproc; ny=Ny; nz=Nz\nnek=int(math.sqrt(2.0)/3*Nx)\ntime = 0.0\n\nchkSz = 32\nslabs = nx//chkSz')


# Here is the data loading part. It can be chosen to be one of two ways. In case there is a previously cached data in disk, it can be read directly without the need to download from the database. In this case, we supose that the data is stored in numpy array binary data, ".npz". Because of the way pyFFTW works, each component must be stored separately, therefore we initialize vx, vy and vz arrays as appropiate for the FFTW work, which is aligned arrays. Typical loading time from disk range from 20 to 60 seconds. 
# 
# In case the data has not been previously cached, it is necessary to download from the database. Due to constraints in the workings of the database itself, there is a maximum size that can be queried on a single function call, consequently, it is necessary to break the download into several queries, which can be described as follows: For a whole $1024^3$ download, each slab contained in a process is of size $128 \times 1024 \times 1024$, and each query is of size $32 \times 1024 \times 1024$, the data is stored temporarely in a list, then concatenated and than reshaped (via np.transpose) to be properly assigned to the vx, vy and vz vectors. As a precaution, the data is cached as soon as the final vectors are calculated. This notebook only calculates a single time-step, but it can be readly generalized to be used with a list of timesteps instead. Typical download times range from 170 to 250 seconds, inside sciserver.

# In[7]:

ipython_shell.run_cell_magic('px', '', '\nvx = ft.zeros_aligned((nx,ny,nz), dtype=\'float32\')\nvy = ft.zeros_aligned((nx,ny,nz), dtype=\'float32\')\nvz = ft.zeros_aligned((nx,ny,nz), dtype=\'float32\')\n\n# Populate velocity field from the Database\n\nif(rank==0):\n    print("Starting the loading process")\n\n##########################################\nload_from_file = False\n\nif(load_from_file):\n    comm.Barrier(); t1=MPI.Wtime()\n    content = np.load(file)\n    if(int(content[\'nproc\'])!=nproc):\n        print("Unmatched number of processes. Must first pre-process to adequate number of process")\n    vx = content[\'vx\']\n    vy = content[\'vy\']\n    vz = content[\'vz\']\n    comm.Barrier(); t2=MPI.Wtime()\n    if(rank==0):\n        print("Finished loading")\n        sys.stdout.write(\'Load from disk: {0:.2f} seconds\\n\'.format(t2-t1))\nelse:\n    comm.Barrier(); t1=MPI.Wtime()\n    lJHTDB = libJHTDB(auth_key)\n    lJHTDB.initialize()\n    ud = []\n    for k in range(slabs):\n        if(rank==0):\n            print(k)\n        start = np.array([rank*nx+k*chkSz, 0, 0],dtype=np.int)\n        width = np.array([chkSz,ny,nz],dtype=np.int)\n        #start = np.array([ 0, 0, rank*nx+k*chkSz],dtype=np.int)\n        ud.append(lJHTDB.getRawData(time,start,width, \n                                    data_set = \'isotropic1024coarse\',\n                                    getFunction = \'Velocity\') )\n    \n    lJHTDB.finalize()\n    comm.Barrier(); t2=MPI.Wtime()\n    if(rank==0):\n        print("Finished loading")\n        sys.stdout.write(\'Load field from database: {0:.2f} seconds\\n\'.format(t2-t1))\n    \n    u = np.concatenate(ud,axis=2)\n    comm.Barrier(); t1=MPI.Wtime()\n    if(rank==0):\n        sys.stdout.write(\'Concatenate: {0:.2f} seconds\\n\'.format(t1-t2))\n    \n    rsh = np.transpose(u,(2,1,0,3))\n    comm.Barrier(); t2=MPI.Wtime()\n    if(rank==0):\n        sys.stdout.write(\'Transpose: {0:.2f} seconds\\n\'.format(t2-t1))\n    ##########################################\n    \n    vx[:,:,:] = rsh[:,:,:,0]\n    vy[:,:,:] = rsh[:,:,:,1]\n    vz[:,:,:] = rsh[:,:,:,2]\n    comm.Barrier(); t1=MPI.Wtime()\n    if(rank==0):\n        sys.stdout.write(\'Splitting: {0:.2f} seconds\\n\'.format(t1-t2))\n        \nif(rank==0):\n    print("vx shape = "+str(vx.shape))')


# Once downloaded, we can cache the data on disk, as being done below. The tipical times required are pretty long, around 1000-1100 seconds, which amounts to 16 to 19 minutes. 

# In[9]:

ipython_shell.run_cell_magic('px', '', '\nfolder = "/home/idies/workspace/scratch"\nfilename = "check-isotropic1024coarse-"+str(rank)+"-(t="+str(time)+")"+".npz"\nfile = folder + "/" + filename\n\ncomm.Barrier(); t1=MPI.Wtime()\nnp.savez(file,vx=vx,vy=vy,vz=vz,nproc=nproc)\ncomm.Barrier(); t2=MPI.Wtime()\nif(rank==0):\n    sys.stdout.write(\'Caching the data: {0:.2f} seconds\\n\'.format(t2-t1))')


# In[ ]:




# In[ ]:



