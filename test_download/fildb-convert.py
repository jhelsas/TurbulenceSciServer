import h5py
#Get mortonz library too.
#On dsp33 load with:
#LD_PRELOAD=/usr/local/myhdf5/lib/libszip.so.2.1:/usr/local/myhdf5/lib/libz.so:/usr/local/myhdf5/lib/libhdf5_hl.so:/usr/local/myhdf5/lib/libhdf5.so python
from numpy import array
import pymorton
import struct
import ctypes

def getblob(x, y, z, blobsize, h5file):
    #definition used to grab a blob in xyz order.
    u = h5file["U"][x:x+blobsize,y:y+blobsize,z:z+blobsize]
    v = h5file["V"][x:x+blobsize,y:y+blobsize,z:z+blobsize]
    w = h5file["W"][x:x+blobsize,y:y+blobsize,z:z+blobsize]
    buffer = ctypes.create_string_buffer(6144) #hardcoded for speed
    index = 0
    offset = 0
    #print ("First val is: " + str(u[0,0,0]))
    for k in range(0, blobsize):
        for j in range(0, blobsize):
            for i in range (0, blobsize):
                #import pdb;pdb.set_trace()
                struct.pack_into('f'*3, buffer, offset,u[k,j,i],v[k,j,i],w[k,j,i])
                offset = offset + 12
    return buffer
#Test it out!

h5file = h5py.File("outpen/OUTPEN_04.h5") #Open for read only
blobsize = 8
#blob = getblob(0,0,0,blobsize, h5file)

f = open('iso4096-102.bin', 'wb+')
zblob = blobsize * blobsize * blobsize
#interleave(x,y,z)
zmax = pymorton.interleave3(1023,511,511)
zmin = pymorton.interleave3(512,0,0)
for zindex in range(zmin,zmax, zblob):
    print ("zindex = " + str(zindex))
    #print ("coord = " + str(pymorton.deinterleave3(zindex)))
    coord = pymorton.deinterleave3(zindex)
    blob = getblob(coord[2], coord[1], coord[0], blobsize, h5file)
    f.write(blob)
    #import pdb;pdb.set_trace()

f.close()
