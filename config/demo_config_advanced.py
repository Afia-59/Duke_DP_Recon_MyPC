"""Demo configuration file."""
import os
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")

import numpy as np

from config import base_config, config_utils
from utils import constants


class Config(base_config.Config):
    """Demo config file.

    Inherit from base_config.Config and override the parameters.
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data_dir = "assets/demo/"  # define directory containing subject image data
        self.subject_id = "test"  # define subject ID
        self.rbc_m_ratio = 0.57  # set RBC to membrane ratio

        # set what kind of segmentation to use
        self.segmentation_key = constants.SegmentationKey.MANUAL_VENT.value
        # define path of manual segmentation
        self.manual_seg_filepath = os.path.join(self.data_dir, "mask.nii")

        # set what kind of hemoglobin correction to use
        self.hb_correction_key = constants.HbCorrectionKey.RBC_AND_MEMBRANE.value
        # define subject hemoglobin in g/dL
        self.hb = 15.0

        # override specific reconstruction parameters, as set in Recon class below
        self.recon = Recon()

        # override reference data, as set in ReferenceData class below
        self.reference_data_key = constants.ReferenceDataKey.MANUAL.value
        self.reference_data = ReferenceData(self.reference_data_key)


class Recon(base_config.Recon):
    """Define reconstruction configurations.

    Attributes:
        recon_key: str, the reconstruction key
        scan_type: str, the scan type
        kernel_sharpness_lr: float, the kernel sharpness for low resolution, higher
            SNR images
        kernel_sharpness_hr: float, the kernel sharpness for high resolution, lower
            SNR images
        n_skip_start: int, the number of frames to skip at the beginning
        n_skip_end: int, the number of frames to skip at the end
        key_radius: int, the key radius for the keyhole image
    """

    def __init__(self):
        """Initialize the reconstruction parameters."""
        super().__init__()
        # override default scan type
        self.scan_type = constants.ScanType.FASTDIXON.value
        # override default number of views to skip
        self.n_skip_start = config_utils.get_n_skip_start(self.scan_type)


class ReferenceData(base_config.ReferenceData):
    """Define reference data.

    Attributes:
        threshold_vent (np.array): ventilation thresholds for binning
        threshold_rbc (np.array): rbc thresholds for binning
        threshold_membrane (np.array): membrane thresholds for binning
        reference_fit_vent (tuple): scaling factor, mean, and std of reference ventilation distribution
        reference_fit_rbc (tuple): scaling factor, mean, and std of reference rbc distribution
        reference_fit_membrane (tuple): scaling factor, mean, and std of reference membrane distribution
        reference_stats (dict): mean and std of defect, low, and high percentage of ventilation,
                                membrane, and rbc reference data
    """

    def __init__(self, reference_data_key):
        super().__init__(reference_data_key)
        # override default ventilation thresholds
        self.threshold_vent = np.array([0.185, 0.418, 0.647, 0.806, 0.933])
        # override default ventilation reference distribution
        self.reference_fit_vent = (0.04074, 0.619, 0.196)


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
