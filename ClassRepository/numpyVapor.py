#!/usr/bin/python2.7
# This will read in a CGNS file and convert it to a vapor VDF fromat
# It needs to call vapor functions so these need to bet set properly
# As environment variables
# ./cgns_to_vdf file.cgns

import os
from array import *
import h5py
import numpy as np
import sys
import struct

def convert_np_scalar_to_vdf(folder,fname,sname,scl,
                             x_f,y_f,z_f,
                             N_x,N_y,N_z,
                             L_x,L_y,L_z):
    '''
    Convert and write numpy array to
    create VFD file from the apor 
    (NCAR) commands
    '''
    
    create_vdf(folder+'/'+fname,[sname],N_x,N_y,N_z,L_x,L_y,L_z)
    scl.astype('float32').tofile(folder+'/'+sname)
    
    populate_vdf(folder,fname,sname) 
        
def create_vdf( file_name, variables,N_x,N_y,N_z,L_x,L_y,L_z):
    ''' 
    Write u array to a binary file and 
    use it to create VFD file from the 
    Vapor (NCAR) commands
    '''

    # minL used to scale dimension so the start at 1.0
    minL=min([L_x,L_y,L_z])
    cmd1=['vdfcreate -dimension '+str(N_x)+'x'+str(N_y) +'x'+str(N_z)+
        ' -level 0 -extents 0:0:0:'+str(L_x/minL)+
        ':'+str(L_y/minL)+':'+str(L_z/minL)+
        ' -vars3d '+':'.join(variables)+' '+ file_name+'.vdf']
        #' -vars3d '+':'.join(variables)+' '+ file_name+'.vdf']
    os.system(cmd1[0])
    print cmd1[0]

def populate_vdf(folder, file_name, var ):
    ''' 
    Populat the VDF file using
    the Vapor (NCAR) commands
    '''
    cmd2=['raw2vdf -varname '+ var +' '+
           (folder+'/'+file_name)+'.vdf ' + (folder+'/'+var)]
    os.system(cmd2[0])
    print cmd2[0]
    
