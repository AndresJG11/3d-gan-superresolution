import nibabel as nib
from deepbrain import Extractor
import numpy as np

import os


PATH = "D:/Datasets/ADNI/MRI"
SAVE_PATH = "D:\Datasets\ADNI\MASK"

files = os.listdir(PATH)

for filename in files:

    file_path = PATH + '/' + filename

    index = filename.split('.')[0]

    print()

    ## proxy = nib.load(file_path)
    ## data = np.array(proxy.dataobj)

    ## print(data.shape)

    ### Load a nifti as 3d numpy image [H, W, D]
    img = nib.load(file_path).get_fdata()

    ext = Extractor()

    ### `prob` will be a 3d numpy image containing probability 
    ### of being brain tissue for each of the voxels in `img`
    prob = ext.run(img) 

    ### mask can be obtained as:
    mask = prob > 0.5

    img = nib.Nifti1Image(np.int16(mask), np.eye(4))  # Save axis for data (just identity)

    img.header.get_xyzt_units()
    img.to_filename(SAVE_PATH + f'/mask_{index}.nii')  # Save as NiBabel file