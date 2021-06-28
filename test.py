# import nibabel as nib
# import numpy as np

from dataset import Train_dataset


traindataset = Train_dataset(1)

# xt_total = traindataset.patches_true(1)

xm_total = traindataset.mask(1)

# import os
# import glob  # For populating the list of files

# path = "D:/Datasets/ADNI/*/*/*/*/*"
# data = glob.glob(path)

# print( os.listdir(path) )



# for folders_patients in os.listdir(path):
#     # os.path.join(path, )
#     print( [ os.path.join(path, folder_patient) for folder_patient in folders_patients] )