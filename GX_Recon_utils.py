from GX_Benchmark_utils import timerfunc  # decorator for timing a function
import platform
import scipy.io as sio
from GX_Twix_parser import readTwix
from numpy.ctypeslib import ndpointer
from datetime import datetime
import re
import os
import ctypes as ct
import pdb
import numpy as np
import scipy.sparse as sps
from scipy.stats import norm
import sys


def generate_radial_1D_traj(dwell_time, grad_delay_time, ramp_time, plat_time, decay_time, npts, oversampling):
    # generate 1d radial distance array based on the timing and the amplitude and the gradient
    grad_delay_npts = grad_delay_time/dwell_time
    ramp_npts = ramp_time/dwell_time
    plat_npts = plat_time/dwell_time
    decay_npts = decay_time/dwell_time

    pts_vec = np.array(range(0, npts))

    # assume a isotropic recon
    def calcRadDist(out_size):
        # calculate sample number of each region boundary
        ramp_start_pt = grad_delay_npts
        plat_start_pt = ramp_start_pt+ramp_npts
        decay_start_pt = plat_start_pt+plat_npts
        decay_end_pt = decay_start_pt+decay_npts

        # calculate binary mask for each region
        in_delay = pts_vec < ramp_start_pt
        in_ramp = (pts_vec >= ramp_start_pt) & (pts_vec < plat_start_pt)
        in_plat = (pts_vec >= plat_start_pt) & (pts_vec < decay_start_pt)
        in_decay = (pts_vec >= decay_start_pt) & (pts_vec < decay_end_pt)

        # calculate times in each region
        ramp_pts_vec = np.multiply((pts_vec-ramp_start_pt), in_ramp)
        plat_pts_vec = np.multiply((pts_vec-plat_start_pt), in_plat)
        decay_pts_vec = np.multiply((pts_vec-decay_start_pt), in_decay)

        # calculate the gradient amplitude  over time(assume plateau is 1)
        ramp_g = ramp_pts_vec/ramp_npts
        plat_g = in_plat
        decay_g = np.multiply((1.0-decay_pts_vec/decay_npts), in_decay)

        # calculate radial position (0.5)
        ramp_dist = 0.5*np.multiply(ramp_pts_vec, ramp_g)
        plat_dist = 0.5*ramp_npts*in_plat + np.multiply(plat_pts_vec, plat_g)
        decay_dist = (0.5*ramp_npts+plat_npts)*in_decay + \
            np.multiply(in_decay, np.multiply(
                decay_pts_vec*0.5, (1.0+decay_g)))

        radial_distance = (ramp_dist+plat_dist+decay_dist)/out_size

        return radial_distance

    return calcRadDist(npts)


def sparse_gridding_c(traj, kernel_para, matrix_size, force_dim):

    # wrap up c function:
    # void sparse_gridding_distance(double *coords, double kernel_width,
    #       unsigned int npts, unsigned int ndims,
    #       unsigned int *output_dims,
    #       unsigned int *n_nonsparse_entries,
    #       unsigned int max_size,
    #       int force_dim)
    # pdb.set_trace()

    if(platform.system() == "Darwin"):
        _sparse = ct.CDLL(os.path.dirname(__file__)+'/libsparse.so')
    elif(platform.system() == "Linux"):
        _sparse = ct.CDLL(os.path.dirname(__file__)+'/libsparse.so')
    else: # windows
        if getattr(sys, 'frozen', False):
            dll_path = os.path.dirname(sys.executable)
        elif __file__:
            dll_path = os.path.dirname(__file__)
        try:
            _sparse = ct.CDLL(dll_path + '/libsparse.dll')
        except:
            _sparse = ct.CDLL(os.path.dirname(__file__)+'/libsparse.so')

    _sparse.sparse_gridding_distance.argtypes = (
        ct.POINTER(ct.c_double), ct.c_double, ct.c_uint, ct.c_uint, ct.POINTER(
            ct.c_uint),
        ct.POINTER(ct.c_uint), ct.c_uint, ct.c_int)

    npts, ndim = np.shape(traj)
    # flatten traj to a list for input
    traj = traj.flatten().tolist()
    kernel_para = kernel_para
    matrix_size = matrix_size.astype(int).flatten().tolist()

    num_coord = len(traj)
    num_matrixsize = len(matrix_size)

    # calculate max size of the output indices
    max_nNeighbors = 1
    for dim in range(0, ndim):
        max_nNeighbors = int(max_nNeighbors*(kernel_para+1))

    max_size = npts*max_nNeighbors

    # create empty output
    nSparsePoints = [0] * 1

    # define argument types
    coord_type = ct.c_double * num_coord
    outputsize_type = ct.c_uint * num_matrixsize
    n_nonsparse_entries_type = ct.c_uint * 1

    # set_result to return an array of numbers. This is deprecated.
    # _sparse.sparse_gridding_distance.restype = ndpointer(
    #     dtype=ct.c_double, shape=(max_size*3,))

    _sparse.sparse_gridding_distance.restype = ct.POINTER(ct.c_double*(max_size*3))

    # originally use c_double in python 2, changed to c_float in python 3 so it will not go out of memory
    # The issue was found caused by Numpy version, need to be below 1.16. We are using 1.14
    # prob ref: https://github.com/numpy/numpy/issues/14214
    result = _sparse.sparse_gridding_distance(
        coord_type(*traj), ct.c_double(kernel_para), ct.c_uint(npts),
        ct.c_uint(ndim), outputsize_type(*matrix_size),
        n_nonsparse_entries_type(*nSparsePoints),
        ct.c_uint(max_size),
        ct.c_int(force_dim)
    )
    # convert pointer to np array
    result = np.asarray(np.ctypeslib.as_array(result.contents, shape=(max_size*3,)))
    sample_indices = result[:max_size]
    voxel_indices = result[max_size:2*max_size]
    distances = result[2*max_size:3*max_size]

    return sample_indices, voxel_indices, distances


def sparse_gridding(traj, kernel_para, matrix_size, force_dim):
    from sparse_gridding_distance import sparse_gridding_distance
    npts, ndim = np.shape(traj)
    max_nNeighbors = 1
    matrix_size = matrix_size.astype(int).flatten().tolist()
    traj = traj.flatten().tolist()
    for dim in range(0, ndim):
        max_nNeighbors = int(max_nNeighbors*(kernel_para+1))
        # create empty output
    nSparsePoints = [0] * 1
    max_size = npts*max_nNeighbors
    result = sparse_gridding_distance(traj, kernel_para, npts, ndim, matrix_size, nSparsePoints, max_size, force_dim)
    sample_indices = result[:max_size]
    voxel_indices = result[max_size:2*max_size]
    distances = result[2*max_size:3*max_size]

    return sample_indices, voxel_indices, distances


def gen_traj_c(num_projs, traj_type):
    # generate xyz coordinates for the trajectory samples based on the number of projs and traj type
    # traj_type:
    # 1: Spiral
    # 2. Halton
    # 3. Haltonized Spiral
    # 4. ArchimedianSeq
    # 5. Double Golden Mean

    # output 3 coordinates of the trajectory points
    output_size = 3*num_projs

    # DM 8/1/19 Replacing this with shared library compiled under Mac OS
    #_traj = ct.CDLL(os.path.dirname(__file__)+'/libtraj.so')
    if(platform.system() == "Darwin"):
        _traj = ct.CDLL(os.path.dirname(__file__)+'/lib_traj.so')
    elif(platform.system() == "Linux"):
        _traj = ct.CDLL(os.path.dirname(__file__)+'/libtraj.so')
    else:
        if getattr(sys, 'frozen', False):
            dll_path = os.path.dirname(sys.executable)
        elif __file__:
            dll_path = os.path.dirname(__file__)
        _traj = ct.CDLL(dll_path + '/libtraj.dll')

    _traj.gen_traj.argtypes = (ct.c_long, ct.c_long)

    # print(os.path.dirname(os.path.abspath(__file__)))
    # print(os.path.dirname(__file__))
    # input()

    # DGM 8/14/19 changing type from double to float to try to save memory does not appear to work. It makes the recon hang. Why!?
    _traj.gen_traj.restype = ndpointer(dtype=ct.c_double, shape=(output_size,))

    result = _traj.gen_traj(ct.c_long(num_projs), ct.c_long(traj_type))

    x = result[:num_projs]
    y = result[num_projs:2*num_projs]
    z = result[2*num_projs:3*num_projs]

    return x, y, z


def read_twix_hdr(bufferr):
    # parse the buffer string into a dictionary
    # remove empty lines
    p = re.compile('\n\s*\n')
    bufferr = p.sub('', bufferr)

    # split ascconv and xprotocco
    p = re.compile('### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###')
    split_list = p.split(bufferr, 1)

    # just take xprot and forget about ascconv
    if(len(split_list) == 1):
        xprot = split_list[0]
        ascconv = []

        # in the case where there is only ascconv
        if('### ASCCONV BEGIN' in xprot[:30]):
            ascconv = xprot
            xprot = []

    elif(len(split_list) == 2):
        # ascconv is not parsed at this moment
        ascconv = split_list[0]
        xprot = split_list[1]
    else:
        raise Exception('Twix file has multiple Ascconv')

    # parse ascconv
    ascconv_dict = {}
    if len(ascconv):

        ascconv = ascconv[22:]  # skip the head: "### Ascconv***"
        p = re.compile('[^=]*=[^=]*\n')
        token_list = p.findall(ascconv)

        for token in token_list:

            name = token.split('=')[0].strip()
            value = token.split('=')[1].strip()

            ascconv_dict[name] = value

    # parse xprot
    twix_dict = {}

    if len(xprot):
        p = re.compile('<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)')
        token_list = p.findall(xprot)

        p = re.compile(
            '<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)')
        token_list = token_list + p.findall(xprot)

        for token in token_list:
            name = token[0]
            p = re.compile('("*)|( *<\w*> *[^\n]*)')
            try:
                value = p.sub(token[-1], '').strip()
            except:
                # print('Key Error in name ' + name)
                continue

            p = re.compile('\s*')
            value = p.sub(value, '')

            twix_dict[name] = value

    return twix_dict, ascconv_dict


def generate_traj(dwell_time, ramp_time, plat_time, decay_time, npts, oversampling, del_x, del_y, del_z, nFrames, traj_type):
    # Generate and vectorize traj and data
    from gen_traj import gen_traj

    traj_para = {
        'npts': npts,
        'dwell_time': dwell_time,
        'oversampling': oversampling,
        'ramp_time': ramp_time,
        'plat_time': plat_time,
        'decay_time': decay_time,
    }

    traj_para.update({'grad_delay_time': del_x})
    radial_distance_x = generate_radial_1D_traj(**traj_para)
    traj_para.update({'grad_delay_time': del_y})
    radial_distance_y = generate_radial_1D_traj(**traj_para)
    traj_para.update({'grad_delay_time': del_z})
    radial_distance_z = generate_radial_1D_traj(**traj_para)

    # x, y, z = gen_traj_c(nFrames, traj_type) # gen_traj.c used

    # use the gen_traj.py
    result = gen_traj(nFrames, traj_type)
    x = result[:nFrames]
    y = result[nFrames:2*nFrames]
    z = result[2*nFrames:3*nFrames]

    x = np.array([radial_distance_x]).transpose().dot(
        np.array([x])).transpose()
    y = np.array([radial_distance_y]).transpose().dot(
        np.array([y])).transpose()
    z = np.array([radial_distance_z]).transpose().dot(
        np.array([z])).transpose()

    return x, y, z

def remove_noise_rays(data, x, y, z, thre_snr, tail=10):
    # remove noisy FID rays in Dixon image
    nFrames = np.shape(data)[0]
    thre_dis = thre_snr*np.average(abs(data[:, :5]))
    print('Afia check: thre_Dis',thre_dis)
    max_tail = np.amax(abs(data[:, tail:]), axis=1)
    good_index = max_tail < thre_dis #afia
    n_Frames_good = np.sum(good_index)
    data = data[good_index]
    x = x[good_index]
    y = y[good_index]
    z = z[good_index]

    return data, x, y, z, n_Frames_good


def complex_align(x):
    return(np.flip(np.flip(np.flip(np.transpose(x, (2, 1, 0)), 0), 1), 2))


def alignrot(x):
    return(np.flip(np.flip(np.flip(x, 0), 1), 2))


def recon(data, traj, kernel_sharpness, kernel_extent, overgrid_factor, image_size, n_pipe_iter, verbosity):

    from GX_Recon_classmap import Gaussian, L2Proximity, MatrixSystemModel, IterativeDCF, LSQgridded

    # Starting reconstruction
    kernel_obj = Gaussian(kernel_extent=kernel_extent,
                          kernel_sigma=kernel_sharpness, verbosity=verbosity)

    prox_obj = L2Proximity(kernel_obj=kernel_obj, verbosity=verbosity)

    system_obj = MatrixSystemModel(proximity_obj=prox_obj, overgrid_factor=overgrid_factor,
                                   image_size=image_size, traj=traj, verbosity=verbosity)

    dcf_obj = IterativeDCF(system_obj=system_obj,
                           dcf_iterations=n_pipe_iter, verbosity=verbosity)

    recon_obj = LSQgridded(system_obj=system_obj,
                           dcf_obj=dcf_obj, verbosity=verbosity)

    reconVol = recon_obj.reconstruct(data=data, traj=traj)
    return(reconVol)
