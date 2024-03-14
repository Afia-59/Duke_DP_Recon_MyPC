# supposed to be a replicate of gen_traj.c
import numpy as np
import math
import sys
import pdb


def halton_number(index, base):
    result = 0
    f = 1.0
    i = index
    while i > 0:
        f = f/base
        result += f*math.fmod(i, base)
        i = int(i/base)
    return result


def haltonSeq(m_adAzimuthalAngle, m_adPolarAngle, num_frames, num_projPerFrame):
    p1 = 2
    p2 = 3
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            z = halton_number(lk+1, p1)*2-1
            phi = 2 * math.pi * halton_number(lk+1, p2)
            m_adPolarAngle[linter] = math.acos(z)
            m_adAzimuthalAngle[linter] = phi


def spiralSeq(m_azi, m_polar, num_frames, num_projPerFrame):
    dPreviousAngle = 0
    num_totalProjections = num_frames * num_projPerFrame
    for lk in range(num_projPerFrame):
        for lFrame in range(num_frames):
            llin = lFrame + lk*num_frames
            linter = lk + lFrame * num_projPerFrame
            dH = -1.0 + 2.0 * llin / float(num_totalProjections)
            m_polar[linter] = math.acos(dH)
            if (llin == 0):
                m_azi[linter] = 0
            else:
                m_azi[linter] = math.fmod(
                    dPreviousAngle+3.6/(math.sqrt(num_totalProjections*(1.0-dH*dH))), 2.0 * math.pi)
            dPreviousAngle = m_azi[linter]


def archimedianSeq(m_azi, m_polar, num_frames, num_projPerFrame):
    dAngle = (3.0-math.sqrt(5.0))*math.pi
    dZ = 2.0/(num_projPerFrame-1.0)

    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame*num_projPerFrame
            m_polar[linter] = math.acos(1.0-dZ*lk)
            m_azi[linter] = lk*dAngle


def dgoldenMSeq(m_azi, m_polar, num_frames, num_projPerFrame):
    goldmean1 = 0.465571231876768
    goldmean2 = 0.682327803828019
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            m_polar[linter] = math.acos(2.0 * math.fmod(lk*goldmean1, 1)-1)
            m_azi[linter] = 2 * math.pi * math.fmod(lk * goldmean2, 1)


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def partition(ht_polar, sp_polar, sp_azi, low, high):
    pivot = ht_polar[high]
    i = low - 1
    for j in range(low, high):
        if ht_polar[j] <= pivot:
            i += 1
            swap(ht_polar, i, j)
            swap(sp_polar, i, j)
            swap(sp_azi, i, j)
    swap(ht_polar, i+1, high)
    swap(sp_polar, i+1, high)
    swap(sp_azi, i+1, high)

    return i+1


def quickSort(ht_polar, sp_polar, sp_azi, low, high):
    if low < high:
        pi = partition(ht_polar, sp_polar, sp_azi, low, high)
        quickSort(ht_polar, sp_polar, sp_azi, low, pi-1)
        quickSort(ht_polar, sp_polar, sp_azi, pi+1, high)


def randomSpiral(m_adAzimuthalAngle, m_adPolarAngle, num_projPerFrame):
    ht_adAzimu = np.zeros(num_projPerFrame)
    ht_adPolar = np.zeros(num_projPerFrame)
    haltonSeq(ht_adAzimu, ht_adPolar, 1, num_projPerFrame)
    quickSort(ht_adPolar, m_adPolarAngle,
              m_adAzimuthalAngle, 0, num_projPerFrame-1)


def gen_traj(m_lProjectionsPerFrame, m_lTrajectoryType):
    m_adAzimuthalAngle = np.zeros(m_lProjectionsPerFrame)
    m_adPolarAngle = np.zeros(m_lProjectionsPerFrame)
    coordinates = np.zeros(m_lProjectionsPerFrame*3)
    m_lNumberOfFrames = 1
    if m_lTrajectoryType == 1:
        spiralSeq(m_adAzimuthalAngle, m_adPolarAngle,
                  m_lNumberOfFrames, m_lProjectionsPerFrame)
    elif m_lTrajectoryType == 2:
        haltonSeq(m_adAzimuthalAngle, m_adPolarAngle,
                  m_lNumberOfFrames, m_lProjectionsPerFrame)
    elif m_lTrajectoryType == 3:
        spiralSeq(m_adAzimuthalAngle, m_adPolarAngle,
                  m_lNumberOfFrames, m_lProjectionsPerFrame)
        randomSpiral(m_adAzimuthalAngle, m_adPolarAngle,
                     m_lProjectionsPerFrame)
    elif m_lTrajectoryType == 4:
        archimedianSeq(m_adAzimuthalAngle, m_adPolarAngle,
                       m_lNumberOfFrames, m_lProjectionsPerFrame)
    else:
        dgoldenMSeq(m_adAzimuthalAngle, m_adPolarAngle,
                    m_lNumberOfFrames, m_lProjectionsPerFrame)
    for k in range(m_lProjectionsPerFrame):
        coordinates[k] = math.sin(m_adPolarAngle[k]) * \
            math.cos(m_adAzimuthalAngle[k])
        coordinates[k+m_lProjectionsPerFrame] = math.sin(
            m_adPolarAngle[k])*math.sin(m_adAzimuthalAngle[k])
        coordinates[k+2*m_lProjectionsPerFrame] = math.cos(m_adPolarAngle[k])
    return coordinates


if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print("Enter [m_lProjectionsPerFrame] [m_lTrajectoryType]")
        raise Exception("Invalid Input")
    elif (len(sys.argv) == 2):
        try:
            m_lProjectionsPerFrame = int(sys.argv[1])
        except ValueError:
            raise Exception("Invalid Input")
        print(gen_traj(m_lProjectionsPerFrame, 3))
    elif (len(sys.argv) == 3):
        try:
            m_lProjectionsPerFrame = int(sys.argv[1])
            m_lTrajectoryType = int(sys.argv[2])
        except ValueError:
            raise Exception("Invalid Input")
        print(gen_traj(m_lProjectionsPerFrame, m_lTrajectoryType))
    else:
        raise Exception("Invalid Input")
