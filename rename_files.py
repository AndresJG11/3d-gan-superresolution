import nibabel as nb
import numpy as np
import glob

import shutil

PATH = "D:/Datasets/ADNI/*/*/*/*/*"
PATH_SAVE = "D:/Datasets/ADNI/MRI"

files = glob.glob(PATH)
i = 0

for file_path in files:

    filename_format = file_path.split("\\")[-1].split(".")

    path_route = "/".join(file_path.split("\\")[:-1])+'/'

    filename = filename_format[0]
    fileformat = filename_format[1]

    file_renamed = f'{i}.' + fileformat

    print(file_renamed)

    # shutil.copyfile(file_path, PATH_SAVE + '/' + file_renamed)

    i += 1
