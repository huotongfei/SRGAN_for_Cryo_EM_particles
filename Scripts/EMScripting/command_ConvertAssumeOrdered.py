
"""
Convert 2 order specified mrcs to npy
"""

__author__ = "Jordy Homing Lam"
__copyright__ = "Copyright 2018, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"


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

import random
import numpy as np
from scipy import spatial
from scipy.spatial.distance import euclidean
from scipy.linalg import eig
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import pickle


from EMScripting_Supporting import *

################################################

parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
parser.add_argument("--UnclearFolder", type=str, dest="unclearfolder",
                    help="The Noisy Raw image")
parser.add_argument("--ClearFolder", type=str, dest="clearfolder",
                    help="The HR image")

args = parser.parse_args()

################################################

# ===================
# I/O
# ===================

# TODO Put as read MRCS pages
def BasicReadMRCs(flist):
    
  mrcarray_clear_page = []
  for fn in flist:
    Voxel = basic_read_mrc(fn)[0]
    for i in range(Voxel.shape[-1]):
        mrcarray_clear_page.append(Voxel[:,:,i])
  return mrcarray_clear_page

# 1. Input as a list of frames 2D
mrcarray_clear_page = BasicReadMRCs(sorted(glob.glob('%s/*.mrcs' %(args.clearfolder))))
mrcarray_unclear_page = BasicReadMRCs(sorted(glob.glob('%s/*.mrcs' %(args.unclearfolder))))

# 2. Output each corresponding page into a np array
MkdirList(['EM_ClearNpy', 'EM_UnclearNpy'])
for i in range(len(mrcarray_clear_page)):
    pickle.dump(mrcarray_clear_page[i], open('EM_ClearNpy/%s.pkl'%(i),'wb'))

for i in range(len(mrcarray_unclear_page)):
    pickle.dump(mrcarray_unclear_page[i], open('EM_UnclearNpy/%s.pkl'%(i),'wb'))
    #fig,ax = plt.subplots(1,1,figsize = (8,8))
    #ax.imshow(np.concatenate((mrcarray_unclear_page[i], mrcarray_clear_page[i]), axis = 0), cmap = "gray")
    #plt.show()



