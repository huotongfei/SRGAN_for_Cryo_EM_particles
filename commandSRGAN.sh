
#=============================================================

# Study on SRGAN applied on EM 2D images

#=============================================================




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
