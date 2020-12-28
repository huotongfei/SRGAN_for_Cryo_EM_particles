

"""
Supporting Code for basic EM applications
"""




import time
import subprocess
import os
import sys
import re
import glob
from io import StringIO
from argparse import ArgumentParser
import shutil
import itertools
from functools import partial
from operator import itemgetter
import gc
import copy
import struct
import numpy as np

# ===================================
# Basic I/O of MRC file
# ===================================
def MkdirList(folderlist): 
 import os
 for i in folderlist:
  if not os.path.exists('%s' %(i)):
   os.mkdir('%s' %(i))


def basic_read_mrc(filename):
    with open(filename, 'rb') as fin:
        MRCdata=fin.read()
        nx = struct.unpack_from('<i',MRCdata, 0)[0]
        ny = struct.unpack_from('<i',MRCdata, 4)[0]
        nz = struct.unpack_from('<i',MRCdata, 8)[0]
        mode = struct.unpack_from('<i',MRCdata, 12)
        mx = struct.unpack_from('<i',MRCdata, 28)[0]
        my = struct.unpack_from('<i',MRCdata, 32)[0]
        mz = struct.unpack_from('<i',MRCdata, 36)[0]

        #side = struct.unpack_from('<f',MRCdata,40)[0]
        a, b, c = struct.unpack_from('<fff',MRCdata,40)
        side = a

        dmin = struct.unpack_from('<fff',MRCdata,76)
        dmax = struct.unpack_from('<fff',MRCdata,80)
        dmean = struct.unpack_from('<fff',MRCdata,84)

        fin.seek(1024, os.SEEK_SET)

        # Density
        rho = np.fromfile(file=fin, dtype=np.dtype(np.float32)).reshape((nx,ny,nz),order='F')
        fin.close()

        return rho, side

def normal_page (imarray):
    min_im = np.min(imarray)
    max_im = np.max(imarray)
    imarray = (imarray - min_im) / (max_im - min_im)
    return imarray

