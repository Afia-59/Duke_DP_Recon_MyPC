"""Demo configuration file."""
import os
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")

from config import base_config
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


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
