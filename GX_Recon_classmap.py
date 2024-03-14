from scipy.stats import norm
import scipy.sparse as sps
import numpy as np
from GX_Benchmark_utils import timerfunc
import pdb
import scipy.io as sio


class Kernel:

    '''Kernel Base (Superclass)'''

    def __init__(self, kernel_extent, verbosity):
        self.verbosity = verbosity
        self.extent = kernel_extent


class Gaussian(Kernel):

    '''Gaussian kernel for gridding'''

    def __init__(self, kernel_extent, kernel_sigma, verbosity):
        # initiate the kernel
        Kernel.__init__(self, kernel_extent=kernel_extent, verbosity=verbosity)
        self.sigma = kernel_sigma
        self.unique_string = 'Gaussian_e'+str(self.extent)+'_s'+str(self.sigma)

    def evaluate(self, kdistance_preovergrid):
        # calculate Nomalized Gaussian Function
        kernel_vals = np.divide(
            norm.pdf(kdistance_preovergrid, 0, self.sigma), norm.pdf(0, 0, self.sigma))

        return kernel_vals

# Proximity class uses the kernel class


class Proximity:

    ''' Proximity Base (Superclass)'''

    def __init__(self, kernel_obj, verbosity):
        self.verbosity = verbosity
        self.kernel_obj = kernel_obj


class L2Proximity(Proximity):

    '''using L2 Proximity for gridding'''

    def __init__(self, kernel_obj, verbosity):
        Proximity.__init__(self, kernel_obj=kernel_obj, verbosity=verbosity)
        self.unique_string = 'L2_'+self.kernel_obj.unique_string

    @timerfunc
    def evaluate(self, traj, overgrid_factor, matrix_size):
        # calculate gridding in pre-overgridding distances
        # with c code compiled
        if(self.verbosity):
            print('Calculating L2 distances ...')

        from GX_Recon_utils import sparse_gridding_c

        sample_idx, voxel_idx, pre_overgrid_distances = sparse_gridding_c(
            traj=traj, kernel_para=overgrid_factor*self.kernel_obj.extent, matrix_size=matrix_size, force_dim=-1)

        pre_overgrid_distances = np.divide(
            pre_overgrid_distances, overgrid_factor)

        if(self.verbosity):
            print('Finished Calculating L2 distances.')
            print('Applying L2 bound ...')

        # look for any values that are out of bound
        keep_values = (sample_idx > 0) & (voxel_idx > 0)
        sample_idx = sample_idx[keep_values]
        voxel_idx = voxel_idx[keep_values]
        pre_overgrid_distances = pre_overgrid_distances[keep_values]

        del keep_values

        if(self.verbosity):
            print('Applying kernel ...')

        kernel_vals = self.kernel_obj.evaluate(pre_overgrid_distances)

        del pre_overgrid_distances

        return sample_idx, voxel_idx, kernel_vals

# System model class uses Proximity class


class SystemModel:

    ''' SystemModel Base (Superclass)'''

    def __init__(self, proximity_obj, overgrid_factor, image_size, verbosity):
        self.verbosity = verbosity
        self.proximity_obj = proximity_obj
        self.overgrid_factor = overgrid_factor
        self.crop_size = np.asarray(image_size)
        self.full_size = np.ceil(np.multiply(
            self.crop_size, self.overgrid_factor)).astype(int)

    def crop(self, uncrop):
        # crop the image
        s_lim = np.round(np.multiply(
            (self.full_size - self.crop_size), 0.5)).astype(int)
        l_lim = np.round(np.multiply(
            (self.full_size + self.crop_size), 0.5)).astype(int)
        return uncrop[s_lim[0]:l_lim[0], s_lim[1]:l_lim[1], s_lim[2]:l_lim[2]]


class MatrixSystemModel(SystemModel):

    '''using Matrix System Model'''

    def __init__(self, proximity_obj, overgrid_factor, image_size, traj, verbosity):
        SystemModel.__init__(self, proximity_obj=proximity_obj,
                             overgrid_factor=overgrid_factor, image_size=image_size, verbosity=verbosity)
        self.unique_string = 'MatMod_'+proximity_obj.unique_string
        self.is_supersparse = False
        self.is_transpose = False

        if(verbosity):
            print('Calculating Matrix interpolation coefficients...')

        sample_idx, voxel_idx, kernel_vals = self.proximity_obj.evaluate(
            traj=traj, overgrid_factor=self.overgrid_factor, matrix_size=self.full_size)
        #
        # np.save('test1',sample_idx)
        # np.save('test2',voxel_idx)
        # np.save('test3',kernel_vals)

        # sample_idx = np.load('test1.npy')
        # voxel_idx = np.load('test2.npy')
        # kernel_vals = np.load('test3.npy')

        if(verbosity):
            print('Finished calculating Matrix interpolation coefficients. :)')

        # making sparse matrix to save storage
        # csr_matrix((data, (row, col)), shape=(3, 3))
        # the -1 is important, converting matlab index to python index
        self.A = sps.csr_matrix((kernel_vals, (sample_idx-1, voxel_idx-1)),
                                shape=(np.shape(traj)[0], np.prod(self.full_size)), dtype=np.float)
        self.A.eliminate_zeros()

        self.ATrans = np.transpose(self.A)

    def makeSuperSparse(self):
        # achieved by eliminate zeros
        return(1)

    def revertSparseness(self, argv):
        # function was used on the old code, not anymore
        return(1)

# Iterative DCF


class IterativeDCF:

    '''Calculate iterative DCF for recon'''

    def __init__(self, system_obj, dcf_iterations, verbosity):
        self.system_obj = system_obj
        self.dcf_iterations = dcf_iterations
        self.verbosity = verbosity
        self.unique_string = 'iter'+str(dcf_iterations)
        self.space = 'dataspace'

        # system_obj is a MatrixSystemModel
        idea_PSFdata = np.ones((system_obj.A.get_shape()[1], 1))

        # reasonable first guess by summing all up
        dcf = np.divide(1, system_obj.A.dot(idea_PSFdata))

        import time
        time_start = time.time()
        # iteratively calculating dcf
        for kk in range(0, self.dcf_iterations):
            if(self.verbosity):
                print(' DCF iteration '+str(kk+1))

            dcf = np.divide(dcf, system_obj.A.dot(system_obj.ATrans.dot(dcf)))

        time_end = time.time()
        print("The runtime for iterative DCF: "+str(time_end-time_start))
        self.dcf = dcf

# LSQ gridded


class GriddedReconModel:

    '''Recon model after gridding'''

    def __init__(self, system_obj, verbosity):
        self.deapodize = False
        self.crop = True
        self.verbosity = verbosity
        self.system_obj = system_obj


class LSQgridded(GriddedReconModel):

    '''LSQ gridding model'''

    def __init__(self, system_obj, dcf_obj, verbosity):
        GriddedReconModel.__init__(
            self, system_obj=system_obj, verbosity=verbosity)
        self.dcf_obj = dcf_obj
        self.unique_string = 'grid_'+system_obj.unique_string+'_'+dcf_obj.unique_string

    @timerfunc
    def grid(self, data):
        # only for MatrixSystemModel
        if(self.dcf_obj.space == 'gridspace'):
            gridVol = np.multiply(
                self.system_obj.ATrans.dot(data), self.dcf_obj.dcf)
        elif(self.dcf_obj.space == 'dataspace'):
            gridVol = self.system_obj.ATrans.dot(
                np.multiply(self.dcf_obj.dcf, data))
        else:
            raise Exception('DCF stype not recognized')
        return(gridVol)

    @timerfunc
    def reconstruct(self, data, traj):
        if(self.verbosity):
            print('Reconstructing ...')
            print('-- Gridding Data ...')

        reconVol = self.grid(data)
        if(self.verbosity):
            print('-- Finished Gridding.')

        reconVol = np.reshape(reconVol, np.ceil(
            self.system_obj.full_size).astype(int))
        if(self.verbosity):
            print('-- Calculating IFFT ...')

        import time
        time_start = time.time()

        reconVol = np.fft.ifftshift(reconVol)
        #KspGas_Afia = np.fft.ifftshift(reconVol)  # afia
        #io.savemat('./tmp/KspGas_Afia.mat', {"KspGas": KspGas_Afia})  # afia
        reconVol = np.fft.ifftn(reconVol)
        reconVol = np.fft.ifftshift(reconVol)

        time_end = time.time()
        print("The runtime for iFFT: "+str(time_end-time_start))

        if(self.verbosity):
            print('-- Finished IFFT.')

        if(self.crop):

            reconVol = self.system_obj.crop(reconVol)
        if(self.deapodize):

            if(self.verbosity):
                print('-- Calculating k-space deapodization function')

            deapVol = self.grid(double(~np.any(traj, axis=1)))

            deapVol = np.reshape(
                deapVol.toarray(), np.ceil(self.system_obj.full_size))

            if(self.verbosity):
                print('-- Calculating image-space deapodization function')

            deapVol = np.ifftn(deapVol)
            deapVol = np.ifftshift(deapVol)

            if(self.crop):
                deapVol = self.system_obj.crop(deapVol)

            reconVol = np.divide(reconVol, deapVol)

            del deapVol

            if(self.verbosity):
                print('-- Finished deapodization.')

        if(self.verbosity):
            print('-- Finished Reconstruction.')

        return(reconVol)
