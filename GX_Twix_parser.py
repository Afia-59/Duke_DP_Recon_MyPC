# Python modules

import xml.etree.cElementTree as ElementTree
import math
import re
import os
import struct
import itertools
import sys
import time

"""This is a simple Python parser for Siemens twix files. It runs under Python
2.5 - 2.7 (and possibly < 2.5). It contains two classes and one function.

The classes are TwixMdh and TwixScan. The former represents the
measurement data header of a single scan from a twix file. The latter is the
same plus the scan data.

The function read() accepts the name of a twix file and returns a list of
TwixScan instances and EVPs.

Brief sample usage is at the end of this file.

This file itself is executable; invoke like so:
   python twix.py  your_filename.dat

Thanks to Assaf Tal, Uzay Emir, and Jonathan Polimeni whose code informed ours.

Enjoy,
Brian Soher and Philip Semanchuk on behalf of the Vespa project
http://scion.duhs.duke.edu/vespa/

==============================
 NEWS
==============================

2012-06-27  Released version 0.1.0, developed and tested only under VB17

2018-01-23  Made it runnable in Python 3.5.2 - Ziyi
"""

VERSION = "0.1.1"





# Basic layout of Twix files are described (briefly!) on p138 of
# IceUsersGuide.pdf


# _MDH_ACQEND is the mask for the first flag in eval_info_mask which indicates
# if this is the last acquisition.
_MDH_ACQEND = 0x01

# There are 2 bytes of DMA flags
_NDMA_FLAG_BYTES = 2
# THere are 8 bytes of eval info flags
_NEVAL_INFO_FLAG_BYTES = 8
# There are 4 ICE params (2 bytes each = 8 bytes)
_NICE_PARAMETERS = 4
# There are 4 free params for pulse sequence use (2 bytes each = 8 bytes)
_NFREE_PARAMETERS = 4


class TwixMdh(object):
    """Represents a Twix MDH (measurement data header). Doesn't contain any
    data.
    """
    def __init__(self):
        # Inline comments are (I think) the names of the corresponding
        # variables in Siemens' mdh.h.

        # Some code treats DMA length as a single 32-bit int and some treats
        # it as a 16-bit in plus two bytes of flags. I'm guessing the former
        # is the easygoing interpretation and that the latter is more correct.
        self.dma_length = 0                     # ulDMALength
        self.dma_flags = [0] * _NDMA_FLAG_BYTES
        self.measurement_userid = 0             # lMeasUID
        self.scan_count = 0                     # ulScanCounter
        # timestamp [2.5 ms ticks since 00:00]
        self.timestamp = 0                      # ulTimeStamp
        # PMU Timestamp, 2.5 ms ticks since last trigger
        self.pmu_timestamp = 0                  # ulPMUTimeStamp

        self.eval_info_mask = [0] * _NEVAL_INFO_FLAG_BYTES  # aulEvalInfoMask

        # Number of complex data points.
        self.samples_in_scan = 0

        # Loop counters. These are 14 unsigned short values which index
        # the measurement (in imaging; not used in spectroscopy).
        self.used_channels = 0                  # ushUsedChannels
        self.line = 0                           # ushLine
        self.acquisition = 0                    # ushAcquisition
        self.slice = 0                          # ushSlice
        self.partition = 0                      # ushPartition
        self.echo = 0                           # ushEcho
        self.phase = 0                          # ushPhase
        self.repetition = 0                     # ushRepetition
        self.set = 0                            # ushSet
        self.segment = 0                        # ushSeg
        # ICE dimension indices
        self.ida = 0                            # ushIda
        self.idb = 0                            # ushIdb
        self.idc = 0                            # ushIdc
        self.idd = 0                            # ushIdd
        self.ide = 0                            # ushIde

        # Cut-off data
        self.pre = 0                            # ushPre
        self.post = 0                           # ushPost

        self.kspace_center_column = 0           # ushKSpaceCentreColumn
        self.coil_select = 0                    # ushCoilSelect
        self.readout_off_center = 0.0           # fReadOutOffcentre
        # Sequence timestamp since last RF pulse
        self.time_since_last_rf = 0             # ulTimeSinceLastRF
        # number of K-space center line
        self.kspace_center_line = 0             # ushKSpaceCentreLineNo
        self.kspace_center_partition = 0        # ushKSpaceCentrePartitionNo
        # 4 ushorts (2 bytes each = 8 total) are reserved for ICE program params
        self.ice_parameters = [0] * _NICE_PARAMETERS
        # 4 ushorts (2 bytes each = 8 total) are free for pulse sequence use
        self.free_parameters = [0] * _NFREE_PARAMETERS

        # Slice data
        self.saggital = 0.0                     # flSag
        self.coronal = 0.0                      # flCor
        self.transverse = 0.0                   # flTra
        self.quaternion = [0.0] * 4             # aflQuaternion

        # doc says, "Channel ID must be the last parameter". Not sure what
        # this means.
        self.channel_id = 0                     # ulChannelId

        # negative, absolute PTAB position in [0.1 mm]
        self.ptab_pos_neg = 0                   # ushPTABPosNeg


    @property
    def is_last_acquisition(self):
        """Returns True if this scan is the last acquisition (which means it
        contains no FID), False otherwise.
        """
        return bool(self.eval_info_mask[0] & _MDH_ACQEND)


    @property
    def clock_time(self):
        """Returns the scan's timestamp as a string representing the clock
        time when the scan was taken. The string is in ISO format and
        includes microseconds -- HH:MM:SS,uSEC

        This function assumes that the scanner's timestamp is in UTC and
        converts it to local time. That assumption might be incorrect. Caveat
        emptor.
        """
        # self.timestamp is the # of 2.5ms ticks since midnight, but when is
        # midnight? Is that in local time or UTC, since many Linux machines
        # use UTC for their system clock?

        # Convert to seconds since midnight.
        timestamp = self.timestamp * 2.5 / 10e4
        microseconds = int(round((timestamp - int(timestamp)) * 10e5))
        timestamp = time.localtime(timestamp)

        # "%H:%M:%S" is ISO "extended" time format (HH:MM:SS)
        # http://en.wikipedia.org/wiki/ISO_8601#Times
        timestamp = time.strftime("%H:%M:%S", timestamp)

        # According to Wikipedia, "Decimal fractions may also be added...
        # A decimal mark, either a comma or a dot...with a preference for a
        # comma according to ISO 8601:2004) is used as a separator..."
        # http://en.wikipedia.org/wiki/ISO_8601#Times
        timestamp += (",%d" % microseconds)

        return timestamp


    def __str__(self):
        return self.__unicode__().encode("utf-8")


    def __unicode__(self):
        lines = [ ]
        lines.append("DMA length:      %d (0x%x)" % (self.dma_length,
                                                self.dma_length))

        lines.append("-- DMA Flags --")
        for i, byte_ in enumerate(self.dma_flags):
            bits = _bit_string(byte_, 8)
            lines.append("   byte[%d]:      0x%02x == %s" % (i, byte_, bits))

        lines.append("Meas userid:     %d" % self.measurement_userid)
        lines.append("Scan count:      %d" % self.scan_count)
        lines.append("Timestamp:       %d" % self.timestamp)
        lines.append("PMU timestamp:   %d" % self.pmu_timestamp)
        lines.append("Clock time:      %s" % self.clock_time)
        lines.append("-- Eval info mask --")
        for i, mask in enumerate(self.eval_info_mask):
            bits = _bit_string(mask, 8)
            lines.append("   mask[%d]:      0x%02x == %s" % (i, mask, bits))

        lines.append("Samples:         %d" % self.samples_in_scan)
        lines.append("Channels used:   %d" % self.used_channels)
        lines.append("Line:            %d" % self.line)
        lines.append("Acquisition:     %d" % self.acquisition)
        lines.append("Slice:           %d" % self.slice)
        lines.append("Partition:       %d" % self.partition)
        lines.append("Echo:            %d" % self.echo)
        lines.append("Phase:           %d" % self.phase)
        lines.append("Repetition:      %d" % self.repetition)
        lines.append("Set:             %d" % self.set)
        lines.append("Segment:         %d" % self.segment)

        lines.append("ICE dim A:       %d" % self.ida)
        lines.append("ICE dim B:       %d" % self.idb)
        lines.append("ICE dim C:       %d" % self.idc)
        lines.append("ICE dim D:       %d" % self.idd)
        lines.append("ICE dim E:       %d" % self.ide)

        lines.append("Pre:             %d" % self.pre)
        lines.append("Post:            %d" % self.post)

        lines.append("K space center column: %d" % self.coil_select)
        lines.append("Coil select:           %d" % self.kspace_center_column)
        lines.append("Readout off center:    %f" % self.readout_off_center)
        lines.append("Time since last RF:    %d" % self.time_since_last_rf)
        lines.append("K space center line:   %d" % self.kspace_center_line)
        lines.append("K space center part:   %d" % self.kspace_center_partition)

        lines.append("-- ICE parameters --")
        for i, parameter in enumerate(self.ice_parameters):
            lines.append("   param[%d]:  %d (0x%x)" % (i, parameter, parameter))

        lines.append("-- Free parameters --")
        for i, parameter in enumerate(self.free_parameters):
            lines.append("   param[%d]:  %d (0x%x)" % (i, parameter, parameter))

        lines.append("Saggital:      %f" % self.saggital)
        lines.append("Coronal:       %f" % self.coronal)
        lines.append("Transverse:    %f" % self.transverse)
        lines.append("Quaternion:    %f %f %f %f" % tuple(self.quaternion))

        lines.append("Channel ID:    %d " % self.channel_id)
        lines.append("PTAB Position: %d " % self.ptab_pos_neg)

        return '\n'.join(lines)


    def populate_from_file(self, infile):
        """Given an open file or file-like object (like a StringIO instance)
        that's positioned at the first byte of an MDH, populates this TwixMdh
        object from the file. The file pointer is advanced to the end of the
        header.
        """
        self.dma_length = _read_ushort(infile)

        self.dma_flags = [_read_byte(infile) for i in range(_NDMA_FLAG_BYTES)]

        self.measurement_userid = _read_int(infile)
        self.scan_count = _read_uint(infile)
        self.timestamp = _read_uint(infile)
        self.pmu_timestamp = _read_uint(infile)

        self.eval_info_mask = [_read_byte(infile) for i in range(_NEVAL_INFO_FLAG_BYTES)]

        self.samples_in_scan = _read_ushort(infile)
        self.used_channels  = _read_ushort(infile)
        self.line           = _read_ushort(infile)
        self.acquisition    = _read_ushort(infile)
        self.slice          = _read_ushort(infile)
        self.partition      = _read_ushort(infile)
        self.echo           = _read_ushort(infile)
        self.phase          = _read_ushort(infile)
        self.repetition     = _read_ushort(infile)
        self.set            = _read_ushort(infile)
        self.segment        = _read_ushort(infile)
        # ICE dimension indices
        self.ida = _read_ushort(infile)
        self.idb = _read_ushort(infile)
        self.idc = _read_ushort(infile)
        self.idd = _read_ushort(infile)
        self.ide = _read_ushort(infile)

        # Cut-off data
        self.pre  = _read_ushort(infile)
        self.post = _read_ushort(infile)

        self.kspace_center_column   = _read_ushort(infile)
        self.coil_select            = _read_ushort(infile)
        self.readout_off_center     = _read_float(infile)
        self.time_since_last_rf     = _read_uint(infile)
        self.kspace_center_line     = _read_ushort(infile)
        self.kspace_center_partition = _read_ushort(infile)

        self.ice_parameters  = [_read_ushort(infile) for i in range(_NICE_PARAMETERS)]
        self.free_parameters = [_read_ushort(infile) for i in range(_NFREE_PARAMETERS)]

        # Slice data
        self.saggital   = _read_float(infile)
        self.coronal    = _read_float(infile)
        self.transverse = _read_float(infile)
        self.quaternion = [_read_float(infile) for i in range(4)]

        self.channel_id = _read_ushort(infile)

        self.ptab_pos_neg = _read_ushort(infile)


class TwixScan(TwixMdh):
    """This is the same as a TwixMdh object, but with the addition of the
    actual scan data (stored as an ordinary Python list).
    """
    def __init__(self):
        TwixMdh.__init__(self)

        self.data = [ ]


    def __unicode__(self):
        s = TwixMdh.__unicode__(self)
        return ("Data points:     %d\n" % len(self.data)) + s


    def populate_from_file(self, infile, data_file=None):
        """Given an open file or file-like object (like a StringIO instance)
        that's positioned at the first byte of an MDH, populates this TwixScan
        object's data and metadata from the file. The file pointer is advanced
        to the end of the header.

        By default, the data is read from the same file as the MDH. When the
        optional data_file parameter is supplied, this method reads the data
        from that file instead. This is useful when the MDH and data are in
        separate files.
        """
        # Make my base class do its work
        TwixMdh.populate_from_file(self, infile)

        if not data_file:
            data_file = infile

        # Data is in complex #s that are stored as (real, imag) pairs.
        data = _read_float(data_file, 2 * self.samples_in_scan)

        self.data = _collapse_complexes(data)



##################   Public functions start here   ##################



def readTwix(filename):
    """Given the name of a Twix file, reads the global header and all the
    scans in the file.

    Returns a two-tuple of (scans, EVPs) where scans is a list of TwixScan
    objects (one for each scan in the file) and EVPs is a list of 2-tuples.
    The EVP two-tuples are (name, data) and contain the name and data
    associated with the ASCII EVP chunks in the global header. They're returned
    in the order in which they appeared.

    To summarize, the return value is a 2-tuple containing two lists:
    (
        [scan1, scan2, ... scanN],
        [ (name1, data1), (name2, data2) (etc.) ]
    )
    """
    file_size = os.path.getsize(filename)

    infile = open(filename, 'rb')

    # Global header size is stashed in the first 4 bytes. This size includes
    # the size itself. In other words, the size can also be interpreted as
    # an absolute offset from 0 that points to the the start of the first scan.
    # In fact, the doc says, "Skip data of this length for reading the real
    # measurement data sorted as in former versions."
    header_size = _read_uint(infile)

    # Next is # of EVPs (whatever they are)
    nevps = _read_uint(infile)

    evps = [ ]
    while nevps:
        # Each EVP contains the name (as a NULL-terminated C string) followed by
        # the number of bytes (characters) of content followed by the content
        # itself. There's no accomodation made for multibyte characters or
        # anything other than ASCII, AFAICT. Or maybe since all the examples in
        # the manual are Windows-based, we should assume a character set
        # of windows-1252?
        name = _read_cstr(infile)
        size = _read_uint(infile)
        data = _read_byte(infile, size)
        data = ''.join(map(chr, data))
        evps.append( (name, data) )
        nevps -= 1

    # There's a few bytes of padding between the EVPs and the start of the
    # scans. Here we jump over that.
    infile.seek(header_size, os.SEEK_SET)

    # Read the scans one by one until we hit the last (which should be flagged)
    # or run off the end of the file.
    scans = [ ]
    more_scans = True
    index = 0
    while more_scans:
        index += 1
        scan = TwixScan()
        scan.populate_from_file(infile)

        if scan.is_last_acquisition:
            more_scans = False
        else:
            scans.append(scan)

            #print '\n                 Scan %d' % index
            #print '----------------------------------------------'
            #print scan

            # PS - I'm not sure if this is necessary. In the samples I have,
            # the scan.is_last_acquisition flag is set appropriately so I
            # never run off the end of the file.
            if infile.tell() >= file_size:
                more_scans = False

    return scans, evps


# Python 2.5 lacks the bin() builtin which we use when printing a scan as
# a string. Here we cook up a reasonable fascimile.
try:
    bin(42)
except NameError:
    # Apparently we are using Python 2.5

    # _BINARY_MAP maps hex characters to their equivalent in bits
    _BINARY_MAP = { '0' : '0000', '1' : '0001',
                    '2' : '0010', '3' : '0011',
                    '4' : '0100', '5' : '0101',
                    '6' : '0110', '7' : '0111',
                    '8' : '1000', '9' : '1001',
                    'a' : '1010', 'b' : '1011',
                    'c' : '1100', 'd' : '1101',
                    'e' : '1110', 'f' : '1111',
                  }

    def bin(value):
        """Given an int, returns a string representing the bits in the int.
        Mimics the bin() builtin of Python >= 2.6, except that this function
        only handles non-negative numbers.
        """
        # Swiped from here:
        # http://stackoverflow.com/questions/1993834/how-change-int-to-binary-on-python-2-5

        # Convert to a hex string
        value = '%x' % value

        # For each character in the string, substitute the binary value
        value = ''.join([_BINARY_MAP[c] for c in value])

        # Trim left 0s
        value = value.lstrip('0')

        # Prefix with '0b' for compatibility with bin()
        return '0b' + value


##################   "Private" functions start here   ##################

def _bit_string(value, min_length=1):
    """Given an int value, returns a bitwise string representation that is
    at least min_length long.

    For example, read._bit_string(42, 8) returns '0b00101010'.
    """
    # Get value and trim leading '0b'
    value = bin(value)[2:]
    # Pad
    value = value.rjust(min_length, '0')
    return '0b' + value



# The _read_xxx() functions are for reading specific types out of a file. All
# call _read_generic().
import pdb

def _read_generic(source_file, type_, count=1):
    format = '<%d%s' % (count, type_)

    data = source_file.read(struct.calcsize(format))

    values = struct.unpack(format, data)
    if count == 1:
        return values[0]
    else:
        return values


def _read_byte(source_file, count=1):
    return _read_generic(source_file, 'B', count)

def _read_ushort(source_file, count=1):
    return _read_generic(source_file, 'H', count)

def _read_float(source_file, count=1):
    return _read_generic(source_file, 'f', count)

def _read_uint(source_file, count=1):
    return _read_generic(source_file, 'I', count)

def _read_int(source_file, count=1):
    return _read_generic(source_file, 'i', count)

def _read_cstr(source_file):
    # Reads until it hits a C NULL (0x00), returns the string (without the NULL)
    chars = [ ]
    char = source_file.read(1).decode('utf-8')

    while char != '\x00':

        chars.append(char)
        char = source_file.read(1).decode('utf-8')


    return ''.join(chars)


def _collapse_complexes(data):
    """Given a list or other iterable that's a series of (real, imaginary)
    pairs, returns a list of complex numbers. For instance, given this list --
       [a, b, c, d, e, f]
    this function returns --
       [complex(a, b), complex(c, d), complex(e, f)]

    The returned list is a new list; the original is unchanged.
    """
    # The original source of this code is:
    # http://scion.duhs.duke.edu/vespa/project/browser/trunk/common/util/io_.py

    # This code was chosen for speed and efficiency. It creates an iterator
    # over the original list which gets called by izip. (izip() is the same
    # as the builtin zip() except that it returns elements one by one instead
    # of creating the whole list in memory.)
    # It's the fastest method of the 5 or 6 I tried, and I think it is also
    # very memory-efficient.
    # I stole it from here:
    # http://stackoverflow.com/questions/4628290/pairs-from-single-list
    data_iter = iter(data)

    return [complex(r, i) for r, i in zip(data_iter, data_iter)]



if __name__ == "__main__":

    argv = sys.argv

#    if len(argv) < 2:
#        #argv.append('D:\\bsoher\\code\\repository_svn\\sample_data\\siemens_twix_vd13\\meas_MID00099_FID06713_eja_svs_press.dat')
#        argv.append('D:\\bsoher\\code\\repository_svn\\sample_data\\example_siemens_twix_wbnaa\\Sarah_WBNAA2.dat')

    if len(argv) < 2:
        print("Please enter the name of the file you want to read.")
        sys.exit(0)
    else:
        scans, evps = readTwix(argv[1])

        import pdb
        pdb.set_trace()

        # Uncomment the two lines below to print each scan to stdout.
        # for scan in scans:
        #     print scan

        # Uncomment the two lines below to write the EVP data to files.
        for name, data in evps:
            open(name + ".txt", "wb").write(data)
