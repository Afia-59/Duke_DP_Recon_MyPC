"""Module for segmentation model inference.

@author: ZiyiW Now Sup
"""
import os

import numpy as np
from absl import app, flags
from scipy.ndimage import zoom

from models.model_vnet import vnet
from utils import constants, img_utils, io_utils
import pydicom

import matplotlib.pyplot as plt

import skimage.util # -- for image montages

def makeSlide(A):
    ''' displays 3D array as a 2D grayscale image montage'''
    plt.imshow(skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], padding_width=1, fill=0),cmap='gray')
    plt.show()

def read_multi_frame_dicom(file_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # Extract pixel data (frames) from the DICOM file
    dicomImage = []
    for frame_index in range(dicom_data.NumberOfFrames):
        frame_data = dicom_data.pixel_array[frame_index]
        dicomImage.append(frame_data)
    dicomImage=np.array(dicomImage)
    dicomImage = np.transpose(dicomImage, (1,2, 0))
    #dicomImage = np.flip(dicomImage, axis=1)
    return dicomImage

HP=read_multi_frame_dicom('cd /mnt/c/Users/usc9q/Documents/Afia/xenon-gas-exchange-consortium-main/62463474_Xe')
makeSlide(HP)

H=read_multi_frame_dicom('cd /mnt/c/Users/usc9q/Documents/Afia/xenon-gas-exchange-consortium-main/62463496_Proton')
makeSlide(H)

def predict(
    image: np.ndarray,
    erosion: int = 0,
) -> np.ndarray:
    """Generate a segmentation mask from the proton or ventilation image.

    Args:
        image: np.nd array of the input image to be segmented.
        image_type: str of the image type ute or vent.
    Returns:
        mask: np.ndarray of type bool of the output mask.
    """
    # get shape of the image
    img_h, img_w, img_s = np.shape(image)
    # # reshaping image for segmentation
    # if img_h == 64 and img_w == 64:
    #     print("Reshaping image for segmentation")
    #     image = zoom(abs(image), [2, 2, 2])
    # elif img_h == 128 and img_w == 128:
    #     pass
    # else:
    #     raise ValueError("Segmentation Image size should be 128 x 128 x n")
    reshaped_slices=[]
    for slice_idx in range(image.shape[2]):
        reshaped_slice = np.reshape(image[:, :, slice_idx], (128, 128))
        reshaped_slices.append(reshaped_slice)

    # Check if the number of slices is less than 128
    if len(reshaped_slices) < 128:
        # Calculate the number of black slices to be added
        num_black_slices_added = 128 - len(reshaped_slices)
        
        # Add zero-filled slices until the total number of slices reaches 128
        for _ in range(num_black_slices_added):
            reshaped_slices.append(np.zeros((128, 128)))

    # Convert the list of reshaped slices back to a NumPy array
    reshaped_image = np.array(reshaped_slices)
    model = vnet(input_size=(128, 128, 128, 1))
    weights_dir_current = "./models/weights/model_ANATOMY_UTE.h5" 
    model.load_weights(weights_dir_current)
    image = img_utils.standardize_image(reshaped_image)
    image = image[None, ...]
    image = image[..., None]
    mask = model.predict(image)
    # Making mask binary
    mask = mask[0, :, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask < 1] = 0
    # erode mask
    if erosion > 0:
        mask = img_utils.erode_image(mask, erosion)
    return mask.astype(bool)

mask = predict(H)
makeSlide(mask)
# export_path = os.path.join(os.path.dirname(FLAGS.nii_filepath), "mask.nii")
# io_utils.export_nii(image=mask.astype("float64"), path=export_path)
# ~~
#     if image_type == constants.ImageType.VENT.value:
#         model = vnet(input_size=(128, 128, 128, 1))
#         weights_dir_current = "./models/weights/model_ANATOMY_VEN.h5"
#     else:
#         raise ValueError("image_type must be ute or vent")

#     # Load model weights
#     model.load_weights(weights_dir_current)

#     if image_type == constants.ImageType.VENT.value:
#         image = img_utils.standardize_image(reshaped_image)
#     else:
#         raise ValueError("Image type must be ute or vent")
#     # Model Prediction
#     image = image[None, ...]
#     image = image[..., None]
#     mask = model.predict(image)
#     # Making mask binary
#     mask = mask[0, :, :, :, 0]
#     mask[mask > 0.5] = 1
#     mask[mask < 1] = 0
#     # erode mask
#     if erosion > 0:
#         mask = img_utils.erode_image(mask, erosion)
    
#     return mask.astype(bool)


# def main(argv):
#     """Run CNN model inference on ute or vent image."""
#     image = io_utils.import_nii(FLAGS.nii_filepath)
#     image_type = FLAGS.image_type
#     mask = predict(image, image_type)
#     export_path = os.path.join(os.path.dirname(FLAGS.nii_filepath), "mask.nii")
#     io_utils.export_nii(image=mask.astype("float64"), path=export_path)


# if __name__ == "__main__":
#     app.run(main)
