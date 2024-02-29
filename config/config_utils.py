"""Util functions for config files."""
from utils import constants


def get_n_skip_start(scan_type: str) -> int:
    """Get the number of frames to skip at the beginning of the dissolved phase scan.

    Args:
        scan_type: str, the scan type
    Returns:
        the number of frames to skip at the beginning
    """
    if scan_type == constants.ScanType.NORMALDIXON.value:
        return 60
    elif scan_type == constants.ScanType.MEDIUMDIXON.value:
        return 60
    elif scan_type == constants.ScanType.FASTDIXON.value:
        return 200
    else:
        raise ValueError(f"Scan type {scan_type} not recognized.")
