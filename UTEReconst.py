import mapvbvd
from GX_Recon_utils import *
from GX_Recon_UTE import recon_ute_um
from utils import io_utils
twixFile='/mnt/c/Users/usc9q/Documents/Afia/xenon-gas-exchange-consortium-main/assets/XeClinical3/meas_MID00148_FID60439_4_xe_radial_ute.dat'
ute = abs(recon_ute_um(twixFile))
np.save('tmp/MyPtoron.npy', ute)
io_utils.export_nii(np.abs(ute), "tmp/Myproton.nii")
