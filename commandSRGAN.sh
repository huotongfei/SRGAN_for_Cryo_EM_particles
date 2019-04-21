
#=============================================================

# Study on SRGAN applied on EM 2D images

#=============================================================

#__author__ = "Jordy Homing Lam"
#__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
#__license__ = "3-clause BSD"


SCRIPTHOME='Scripts/EMScripting'

# ===================================
# Processing MRCS into Numpy Array
# ===================================

#python ${SCRIPTHOME}/command_ConvertAssumeOrdered.py  --UnclearFolder EM_Unclear --ClearFolder EM_Clear


# ====================================
# Training SRGAN
# ====================================

python ${SCRIPTHOME}/command_TrainSRGAN.py --UnclearFolder EM_UnclearNpy --ClearFolder EM_ClearNpy


# ====================================
# Testing SRGAN
# ====================================







# ====================================
# Visualise SR image
# ====================================
