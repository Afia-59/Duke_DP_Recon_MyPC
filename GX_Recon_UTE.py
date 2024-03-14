from GX_Recon_utils import *
import numpy as np
import mapvbvd
from scipy.ndimage import zoom

def reconstructUTE(twix_obj, data, orientation):
    """This Function will be used to reconstruct all the UTE data of any site and 
    sequence"""
    dwell_time = twix_obj.hdr.Phoenix[('sRXSPEC', 'alDwellTime', '0')]
    dwell_time = float(dwell_time)/1000.0
    
    # Extracting the Ramp time 
    try:
        ramp_time = twix_obj['hdr']['Meas']['RORampTime']
        if ramp_time == 0.0: ramp_time = 100.0
    except:
        ramp_time = float(twix_obj['hdr']['Meas']['alRegridRampupTime'].split()[0])    
        if ramp_time == 0.0: 
            ramp_time = 100.0
        else:
            ramp_time = 100.0
    print('ramp time' + str(ramp_time))
    # Assuming Siemens Scanners are correcting for gradient delay
    grad_delay_x = -5; grad_delay_y = -5; grad_delay_z = -5

    npts = data.shape[1]
    nFrames = data.shape[0]

    gen_traj_dict = {
        'npts': npts,
        'nFrames': nFrames,
        'traj_type': 3,  # halton Spiral
        'dwell_time': dwell_time,
        'oversampling': 3,
        'ramp_time': ramp_time,
        'plat_time': 2500,
        'decay_time': 60,
        'del_x': grad_delay_x,
        'del_y': grad_delay_y,
        'del_z': grad_delay_z,
    }

    x, y, z = generate_traj(**gen_traj_dict)

    def vectorize(x):
        return np.reshape(x, (np.prod(np.shape(x)), 1))

    traj = np.squeeze(
        0.5*np.stack((vectorize(x), vectorize(y), vectorize(z)), axis=-1))

    kernel_sharpness = 0.15
    kernel_extent = 7*kernel_sharpness

    traj_scale = npts/128.0

    # Deciding Reconstruction Factors for different resolution data. 
    if npts == 128.0: 
        recon_factor = 1
    elif npts == 64.0:
        recon_factor = 2
    else:
        pass
    wd = os.getcwd() #afia
    trajUTE_afia=traj
    np.save(os.path.join(wd, "trajUTE_afia"), trajUTE_afia)  # afia
    recon_dict = {
        'traj': traj*traj_scale,
        'data': np.reshape(data, (npts*nFrames, 1)),
        'kernel_sharpness': kernel_sharpness,
        'kernel_extent': kernel_extent,
        'overgrid_factor': 3,
        'n_pipe_iter': 20,
        'image_size': (recon_factor*npts, recon_factor*npts, recon_factor*npts),
        'verbosity': 0,
    }
    print('Starting recon UTE')
    uteVol = recon(**recon_dict)
    
    # Image Rotation, according to the acquisition, Transversal or Coronal
    def complex_rot_axial(x):
        from scipy.ndimage import rotate
        real = rotate(rotate(np.real(x),90,(1,2)),270)
        imag = rotate(rotate(np.imag(x),90,(1,2)),270)
        return real + 1j*imag

    if orientation == 'Transversal':
        uteVol = complex_rot_axial(complex_align(uteVol))
    elif orientation == 'Coronal':
        uteVol = complex_align(uteVol)
    else: 
        pass

    # # Resizing the images to 128 by 128, if needed
    # def interp(img, factor):
    #     """This function interpolates the images to make the dimension 128 by 128"""
    #     img_real = zoom(np.real(img), [factor, factor, factor])
    #     img_imag = zoom(np.imag(img), [factor, factor, factor])
    #     return img_real + 1j * img_imag

    # if npts != 128:
    #     factor = 128/npts;
    #     uteVol = interp(uteVol, factor)
    # else:
    #     pass

    return uteVol

def recon_ute_duke(twix_file):
    '''This function will reconstruct all the Duke UTE data'''

    # Reading the raw twix file
    twix_obj_all = mapvbvd.mapVBVD(twix_file)
    # The twix files can contain a separate noise field. 
    # This try except block will allow to process old and new
    # duke data.
    try:
        twix_obj_all.image.squeeze = True
        twix_obj_all.image.flagIgnoreSeg = True
        twix_obj_all.image.flagRemoveOS = False
        data_all = twix_obj_all.image.unsorted()
        twix_obj = twix_obj_all
    except:
        twix_obj = twix_obj_all[1]  
        twix_obj.image.squeeze = True
        twix_obj.image.flagIgnoreSeg = True
        twix_obj.image.flagRemoveOS = False
        data_all = twix_obj.image.unsorted()

    if data_all.ndim == 3:
        data_dummy = np.transpose(np.squeeze(data_all[:,0,:]))
        data = data_dummy[0:4600,:] # Excluding last 30 cali FIDs, Number 4601 has no signal
    else:
        data = data_all.T  # transposing the array

    try:
        # Trying to get the acquisition plane
        orientation = twix_obj.hdr.Dicom.tOrientation
    except:
        # If not present, hard-coding it to coronal
        orientation = 'Coronal'

    # Reconstructing the raw UTE data
    uteVol = reconstructUTE(twix_obj, data, orientation = orientation)
    
    return(uteVol)

def recon_ute_uva(twix_file):
    # Reading all the data
    twix_obj_all = mapvbvd.mapVBVD(twix_file)
    try:
        twix_obj_all.image.squeeze = True
        twix_obj_all.image.flagIgnoreSeg = True
        twix_obj_all.image.flagRemoveOS = False
        data_all = twix_obj_all.image.unsorted()
        twix_obj = twix_obj_all
    except:
        twix_obj = twix_obj_all[1]  
        twix_obj.image.squeeze = True
        twix_obj.image.flagIgnoreSeg = True
        twix_obj.image.flagRemoveOS = False
        data_all = twix_obj.image.unsorted()

    if data_all.ndim == 3:
        data = np.transpose(np.squeeze(data_all[:,0,:]))
        #data = data2[0:4600,:] # Excluding last 30 cali FIDs, Number 4601 has no signal
    else:
        data = data_all.T

    # Reconstructing the raw UTE data
    uteVol = reconstructUTE(twix_obj, data, orientation = 'Coronal')

    return(uteVol)

def recon_ute_sickkids(twix_file):
    # Reading all the data
    twix_obj = mapvbvd.mapVBVD(twix_file)
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    # Reading raw Image data
    data_all  = twix_obj.image.unsorted()
    data = np.transpose(data_all[:,0,:]) # Dimension 0 has more signal and contrast
    
    # Reconstructing the raw UTE data
    uteVol = reconstructUTE(twix_obj, data, orientation = 'Coronal')
    
    return(uteVol)

def recon_ute_um(twix_file):
    '''This function will reconstruct all the Duke UTE data'''

    # Reading the raw twix file
    twix_obj_all = mapvbvd.mapVBVD(twix_file)
    # The twix files can contain a separate noise field. 
    # This try except block will allow to process old and new
    # duke data.
    try:
        twix_obj_all.image.squeeze = True
        twix_obj_all.image.flagIgnoreSeg = True
        twix_obj_all.image.flagRemoveOS = False
        data_all = twix_obj_all.image.unsorted()
        twix_obj = twix_obj_all
    except:
        twix_obj = twix_obj_all[1]  
        twix_obj.image.squeeze = True
        twix_obj.image.flagIgnoreSeg = True
        twix_obj.image.flagRemoveOS = False
        data_all = twix_obj.image.unsorted()

    if data_all.ndim == 3:
        data_dummy = np.transpose(np.squeeze(data_all[:,0,:]))
        data2 = data_dummy[0:7200,:] # Excluding last 30 cali FIDs, Number 4601 has no signal
        data = data2[::2,:]
    
    else:
        data = data_all.T  # transposing the array

    try:
        # Trying to get the acquisition plane
        orientation = twix_obj.hdr.Dicom.tOrientation
    except:
        # If not present, hard-coding it to coronal
        orientation = 'Coronal'

    # Reconstructing the raw UTE data
    uteVol = reconstructUTE(twix_obj, data, orientation = orientation)
    
    return(uteVol)

if __name__ == '__main__':
    from GX_Map_utils import export_nii

    if len(sys.argv) == 1:
        print("Enter UTE file path")
        filepath = input()
    elif len(sys.argv) == 2:
        filepath = sys.argv[1]

    uteVol = recon_ute_duke(filepath)
    export_path = os.path.dirname(filepath) + '/ute.nii'
    export_nii(abs(uteVol).astype('float64'), path=export_path)