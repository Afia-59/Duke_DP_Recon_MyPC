import os
from typing import Tuple

import nibabel as nib
import numpy as np
from absl import app, flags
from scipy import signal

from utils import constants, img_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("image_file", "", "nifti image file path.")
flags.DEFINE_string("mask_file", "", "nifti mask file path.")
flags.DEFINE_string("output_path", "", "output folder location")

current_path=os.getcwd()
nii_img = nib.load(os.path.join(current_path,"tmp/image.nii"))
image = nii_img.get_fdata()

nii_img = nib.load(os.path.join(current_path,"tmp/mask.nii"))
mask = nii_img.get_fdata()

def correct_biasfield_n4itk(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply N4ITK bias field correction.

    Args:
        image: np.ndarray 3D image to apply n4itk bias field correction.
        mask: np.ndarray 3D mask for n4itk bias field correcton.
    """
    current_path = os.path.dirname(__file__)
    tmp_path = os.path.join(current_path, "tmp")
    bin_path = os.path.join(current_path, "bin")

    pathInput = os.path.join(tmp_path, "image.nii")
    pathMask = os.path.join(tmp_path, "mask.nii")
    pathOutput = os.path.join(tmp_path, "image_cor.nii")
    pathBiasField = os.path.join(tmp_path, "biasfield.nii")

    pathN4 = os.path.join(bin_path, "N4BiasFieldCorrection")
    # save the inputs into nii files so the execute N4 can read in
    nii_imge = nib.Nifti1Image(np.abs(image), np.eye(4))
    nii_mask = nib.Nifti1Image(mask.astype(float), np.eye(4))
    nib.save(nii_imge, pathInput)
    nib.save(nii_mask, pathMask)
    cmd = (
        pathN4
        + " -d 3 -i "
        + pathInput
        + " -s 2 -x "
        + pathMask
        + " -o ["
        + pathOutput
        + ", "
        + pathBiasField
        + "]"
    )

    os.system(cmd)

    image_cor = np.array(nib.load(pathOutput).get_fdata())
    image_biasfield = np.array(nib.load(pathBiasField).get_fdata())

    # remove the generated nii files
    os.remove(pathInput)
    os.remove(pathMask)
    os.remove(pathOutput)
    os.remove(pathBiasField)

    return image_cor.astype("float64"), image_biasfield.astype("float64")
