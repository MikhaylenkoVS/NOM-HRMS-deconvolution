""" THE PYMSFILEREADER MODULE

Author: Fat Forest Carp

===== CAUTION =====
In order to use this module, the MSFileReader library must be intalled
on your computer. The installer can be downloaded from the official website:
https://thermo.flexnetoperations.com/control/thmo/search?query=MSFileReader&search=Search&sortby=rank
"""
from ctypes import *
import math

import comtypes
import comtypes.client
from comtypes.automation import VARIANT

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 7})


class PyMSFileReader:
    """ Uses the MSFileReader interface for working with a RAW file.

    This class wraps the specified RAW file and provides an interface to
    interact with it. 

    Note
    ----
    * "Spectrum list" is a "peak list". The term "spectrum list" is used in
      XCalibur.
    * Thermo RAW file consists of a chromatogram and a collection of
      mass-specta (MS SCANS). Each MS scan has a unique scan number (index)
      and retention time.
    * SEGMENTS are MS scans that differ in minimum/maximum mz value and
      scan type (Full, SIM, Zoom, SRM). All segments are separated successive
      MS scans. Google 'Segment Scan Mass Spectral Acquisition' for more
      information.
    * If the presence of segment mass spectrum acquisition, the mass
      spectrometer sequentially measures several types of segments along the
      chromatogram. For example, segments with mz 90-310, mz 290-510, and
      mz 490-710. After measuring a segment of mz 490-710, the next segment
      will again be with mz 90-310, and the process repeats So, this
      acquision technique divides the chromatogram into consecutive BLOCKS:
          [mz 90 - 310, mz 290 - 510, mz 490 - 710],
          [mz 90 - 310, mz 290 - 510, mz 490 - 710],
          ...

    Attributes
    ----------
    xraw : comtypes.POINTER
        MSFileReader COM interface for the specified RAW file.
    is_segmented : bool
        Checks if the RAW file consists of different types of segments.
    segment_indices : dict[str, list]
        Collection of scan indices grouped by type of segment. Not used if
        'is_segmented' is False.
    segment_blocks : list[list] or None
        List of blocks. This variable is not used if 'is_segmented' equals
        False. A block is a collection of successive indices.
    """

    def __init__(self,
                 raw_file_path: str):
        """ A constructor.

        Parameters
        ----------
        raw_file_path : str
            Path to a RAW file.
        """
        self.xraw = comtypes.client.CreateObject('MSFileReader.XRawFile.1')
        self.xraw.Open(raw_file_path)
        self.xraw.SetCurrentController(0, 1)

        # Noise removal ------------------------------------------------------
        self._intensity_step = 0.1
        self._intensity_ceiling = 50

        # Collect and classify segments --------------------------------------
        total = c_long()
        self.xraw.GetNumSpectra(total)

        self.segment_indices = {}

        for i in range(total.value + 1):
            scan_info = self._get_info_from_scan_num(c_long(i))
            scan_type = self._get_scan_type(c_long(i))

            # ignore magic 0th scan
            if scan_info["pdHighMass"] == 0: 
                continue

            key = f'{scan_info["pdLowMass"]}-{scan_info["pdHighMass"]}__'\
                  f'{scan_type.value}'

            if key not in self.segment_indices:
                self.segment_indices[key] = []

            self.segment_indices[key].append(i)

        self.is_segmented = len(self.segment_indices.keys()) > 1

        # Collect blocks of segments -----------------------------------------
        self.segment_blocks = None

        if self.is_segmented:
            self.segment_blocks = list(
                zip(*[self.segment_indices[k] for k
                      in list(self.segment_indices.keys())])
            )

    # ------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------
    def get_mass_list_from_SCAN_NUM(self,
                                    pnScanNumber: c_long) -> np.ndarray:
        """ Returns a mass spectum for a given scan number.

        A mass spectrum contains all m/z values, not just for peaks.

        Parameters
        ----------
        scan_num : c_long
            A scan number.

        Returns
        -------
        data : numpy.ndarray
            A mass spectrum (N, 6).
        """
        # Inputs
        szFilter              = ""
        nIntensityCutoffType  = c_long(0)  # None
        nIntensityCutoffValue = c_long(0)
        nMaxNumberOfPeaks     = c_long(0)  # All data points
        bCentroidResults      = False
        nCentroidPeakWidth    = c_double(0)

        # Outputs
        data = np.array('F')
        pvarMassList = VARIANT()
        pvarFlags = VARIANT()
        pnArraySize = c_long(0)

        result = self.xraw.GetMassListFromScanNum(
                pnScanNumber,
                szFilter,
                nIntensityCutoffType,
                nIntensityCutoffValue,
                nMaxNumberOfPeaks,
                bCentroidResults,
                nCentroidPeakWidth,
                pvarMassList,
                pvarFlags,
                byref(pnArraySize)
        )

        if result != 0:
            print(f'Error code from GetMassListFromScanNum: {result}',
                  '- see the Reference guide!')
            quit()

        data = np.array(pvarMassList.value).transpose()
        # data = data[:, :2]

        return data

    def get_spectrum_list_from_SCAN_NUM(self,
                                        pnScanNumber: c_long) -> np.ndarray:
        """ Returns a spectum list for a given scan number.

        Returns one non-averaged spectrum list.

        Parameters
        ----------
        scan_num : c_long
            A scan number.

        Returns
        -------
        data : numpy.ndarray
            A spectrum list (N, 6).
        """
        # Outputs
        data = np.array('F')
        # Inputs
        pvarLabels = VARIANT()
        pvarFlags = VARIANT()

        result = self.xraw.GetLabelData(pvarLabels, pvarFlags, pnScanNumber)

        if result != 0:
            print(f'Error code from GetLabelData: {result}',
                  '- see the Reference guide!')
            quit()

        data = np.array(pvarLabels.value).transpose()
        # data = data[:, :2]

        return data

    def get_averaged_spectrum_list_from_SCAN_NUMLIST(self,
            scan_indices: list) -> np.ndarray:
        """ Returns a spectum list averaged from several scans.

        Note
        ----
        Use it only for unsegmented RAW files.

        Parameters
        ----------
        scan_indices : list

        Returns
        -------
        data : numpy.ndarray
            A spectrum list (N, 6).
        """

        if self.is_segmented:
            print('Use the method',
                  '`get_averaged_spectrum_list_from_SCAN_NUMLIST`',
                  'only for unsegmented RAW files.')
            return None

        scan_indices = [0] + scan_indices  # magic 0th scan
        c_long_array = c_long * len(scan_indices)

        # Outputs
        data = np.array('F')
        # Inputs
        pnScanNumbers = c_long_array(*scan_indices)
        nScansToAverage = c_long(len(scan_indices))
        pvarMassList = VARIANT()
        pvarPeakFlags = VARIANT()
        pnArraySize = c_long()

        res = self.xraw.GetAveragedLabelData(pnScanNumbers,
                                             nScansToAverage,
                                             pvarMassList,
                                             pvarPeakFlags,
                                             pnArraySize)

        if res != 0:
            print(f'Error code from GetAveragedLabelData: {res}',
                  '- see the Reference guide!')
            quit()

        data = np.array(pvarMassList.value).transpose()
        # data = data[:, :2]

        return data

    def get_averaged_spectrum_list_from_FILE(self) -> dict:
        """ Returns a spectum list averaged from ALL scans.

        Note
        ----
        Averages scans of the same type together.
        If there are no scans, return a dict with one item.

        Returns
        -------
        data : dict[str, numpy.ndarray]
            A collection of spectrum lists (N, 6) for each segment type.
        """
        result = {}

        for scan_type, scan_indices in self.segment_indices.items():

            scan_indices = [0] + scan_indices  # magic 0th scan
            c_long_array = c_long * len(scan_indices)

            # Outputs
            data = np.array('F')
            # Inputs
            pnScanNumbers = c_long_array(*scan_indices)
            nScansToAverage = c_long(len(scan_indices))
            pvarMassList = VARIANT()
            pvarPeakFlags = VARIANT()
            pnArraySize = c_long()

            res = self.xraw.GetAveragedLabelData(pnScanNumbers,
                                                 nScansToAverage,
                                                 pvarMassList,
                                                 pvarPeakFlags,
                                                 pnArraySize)

            if res != 0:
                print(f'Error code from GetAveragedLabelData: {res}',
                      '- see the Reference guide!')
                quit()

            data = np.array(pvarMassList.value).transpose()
            # data = data[:, :2]

            result[scan_type] = data

        return result

    def get_averaged_spectrum_list_from_RT(self,
                                           start_rt: c_double,
                                           end_rt: c_double) -> dict:
        """ Returns a spectum list averaged from several scans.

        This function takes into account scans whose retention time is between
        'start_rt' and 'finish_rt' (This is approximately because
        'ScanNumFromRT' looks for the scan num with the closest retention
        time).

        Parameters
        ----------
        start_rt : c_double
            The minimum retention time.
        finish_rt : c_double
            The maximum retention time.

        Returns
        -------
        data : numpy.ndarray
            A spectrum list (N, 6).
        """
        start_scan_num = c_long()
        self.xraw.ScanNumFromRT(start_rt, start_scan_num)

        end_scan_num = c_long()
        self.xraw.ScanNumFromRT(end_rt, end_scan_num)

        # Include an integer number of blocks if the file contains segments
        if self.is_segmented:
            start_scan_num, end_scan_num = self._align_with_block_borders(
                    start_scan_num.value,
                    end_scan_num.value
            )
        else:
            start_scan_num = start_scan_num.value
            end_scan_num = end_scan_num.value

        result = {}

        # Average segments of the same type between RT1 and RT2
        for scan_type, scan_indices in self.segment_indices.items():

            indices = self._get_slice_of_indices(start_scan_num,
                                                 end_scan_num,
                                                 scan_indices)

            indices.insert(0, 0)  # magic 0th scan
            c_long_array = c_long * len(indices)

            # Outputs
            data = np.array('F')
            # Inputs
            pnScanNumbers = c_long_array(*indices)
            nScansToAverage = c_long(len(indices))
            pvarMassList = VARIANT()
            pvarPeakFlags = VARIANT()
            pnArraySize = c_long()

            res = self.xraw.GetAveragedLabelData(pnScanNumbers,
                                                 nScansToAverage,
                                                 pvarMassList,
                                                 pvarPeakFlags,
                                                 pnArraySize)

            if res != 0:
                print(f'Error code from GetAveragedLabelData: {res}',
                      '- see the Reference guide!')
                quit()

            data = np.array(pvarMassList.value).transpose()
            # data = data[:, :2]

            result[scan_type] = data

        return result

    def get_min_rt(self) -> c_double:
        """ Returns the minimum retention time from the given file. """
        min_rt = c_double()
        self.xraw.GetStartTime(min_rt)

        return min_rt

    def get_max_rt(self) -> c_double:
        """ Returns the maximum retention time from the given file. """
        max_rt = c_double()
        self.xraw.GetEndTime(max_rt)

        return max_rt

    def get_last_spectrum_number(self) -> c_long:
        """ Returns the number of scans in the given file. """
        last_n = c_long()
        self.xraw.GetLastSpectrumNumber(last_n)

        return last_n

    def save_csv_file(self,
                      file_path: str,
                      data: np.ndarray,
                      all_data: bool=False) -> None:
        """ Saves data to a CSV file.

        Parameters
        ----------
        file_path : str
            A path to a file.
        data : numpy.ndarray
            A spectrum list.
        """
        with open(file_path, 'w') as wf:
            if not all_data:
                wf.write('m/z,Intensity\n')
                for pair in data:
                    wf.write(f'{self._round_half_up(pair[0], 12)},'
                             f'{self._round_half_up(pair[1], 12)}\n')
            else:
                for pair in data:
                    wf.write(f'{self._round_half_up(pair[0], 12)},'
                             f'{self._round_half_up(pair[1], 12)},'
                             f'{self._round_half_up(pair[2], 5)},'
                             f'{self._round_half_up(pair[3], 5)},'
                             f'{self._round_half_up(pair[4], 5)},'
                             f'{self._round_half_up(pair[5], 5)}\n')

    def save_tab_separated_file(self,
                                file_path: str,
                                data: np.ndarray,
                                all_data: bool=False) -> None:
        """ Saves data to a tab separated TXT file.

        Parameters
        ----------
        file_path : str
            A path to a file.
        data : numpy.ndarray
            A spectrum list.
        """
        if not all_data:
            data = data[:, :2]

        with open(file_path, 'w') as wf:
            if not all_data:
                wf.write('m/z\tIntensity\n')
                for pair in data:
                    wf.write(f'{self._round_half_up(pair[0], 12)}\t'
                             f'{self._round_half_up(pair[1], 12)}\n')
            else:
                wf.write('m/z\tIntensity\tResolution\tBaseline\t'
                         'Noise\tCharge\n')
                for pair in data:
                    wf.write(f'{self._round_half_up(pair[0], 12)}\t'
                             f'{self._round_half_up(pair[1], 12)}\t'
                             f'{self._round_half_up(pair[2], 5)}\t'
                             f'{self._round_half_up(pair[3], 5)}\t'
                             f'{self._round_half_up(pair[4], 5)}\t'
                             f'{self._round_half_up(pair[5], 5)}\t')


    # ------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------
    def _align_with_block_borders(self,
                                  start_idx: int,
                                  end_idx: int) -> tuple:
        """ Corrects the start and end indices of a range.

        This function ensures that the program processes an integer number of
        blocks.

        Returns
        -------
        tuple(int, int)
            Corrected the start and end indices values.
        """
        for block in self.segment_blocks:
            if start_idx in block:
                start_idx = block[0]

            if end_idx in block:
                end_idx = block[-1]

        return start_idx, end_idx

    def _get_info_from_scan_num(self,
                                nScanNumber: c_long) -> dict:
        """ Returns the header info values for the given scan number.
        
        Parameters
        ----------
        nScanNumber : c_long
            The number of the scan.
        
        Returns
        -------
        dict
            Dictionary with the info variables.
        """
        # Outputs
        pnNumPackets = c_long()
        pdStartTime = c_double()
        pdLowMass = c_double()
        pdHighMass = c_double()
        pdTIC = c_double()
        pdBasePeakMass = c_double()
        pdBasePeakIntensity = c_double()
        pnNumChannels = c_long()
        pbUniformTime = c_long()
        pdFrequency = c_double()

        self.xraw.GetScanHeaderInfoForScanNum(nScanNumber,
                                              pnNumPackets,
                                              pdStartTime,
                                              pdLowMass,
                                              pdHighMass,
                                              pdTIC,
                                              pdBasePeakMass,
                                              pdBasePeakIntensity,
                                              pnNumChannels,
                                              pbUniformTime,
                                              pdFrequency)

        scan_info = {
            "pnNumPackets": pnNumPackets.value,
            "pdStartTime": pdStartTime.value,
            "pdLowMass": pdLowMass.value,
            "pdHighMass": pdHighMass.value,
            "pdTIC": pdTIC.value,
            "pdBasePeakMass": pdBasePeakMass.value,
            "pdBasePeakIntensity": pdBasePeakIntensity.value,
            "pnNumChannels": pnNumChannels.value,
            "pbUniformTime": pbUniformTime.value,
            "pdFrequency": pdFrequency.value,
        }

        return scan_info

    def _get_scan_type(self, nScanNumber: c_long) -> c_long:
        """ Returns scan type for the given scan number.
        
        Possible values:
            0 - Full;
            1 - SIM;
            2 - Zoom;
            3 - SRM.
        """
        pnScanType = c_long()
        self.xraw.GetScanTypeForScanNum(nScanNumber, pnScanType)

        return pnScanType

    @staticmethod
    def _get_slice_of_indices(global_start_idx: int,
                              global_end_idx: int,
                              indices_of_this_type: list) -> list:
        """ Finds which indices for this scan type to use.
        
        Parameters
        ----------
        global_start_idx : int
            Global start index.
        global_end_idx : int
            Global end index.
        indices_of_this_type : list
            List of scan indices belonging to a specific scan type.

        Returns
        -------
        list
            List of scans of this type which satisfies the global indices.
        """
        slice_start_idx = 0
        slice_end_idx = len(indices_of_this_type) - 1

        for i in range(len(indices_of_this_type)):
            if indices_of_this_type[i] >= global_start_idx:
                slice_start_idx = i
                break

        for i in range(len(indices_of_this_type)):
            if indices_of_this_type[i] > global_end_idx:
                slice_end_idx = i - 1
                break

        return indices_of_this_type[slice_start_idx: slice_end_idx + 1]

    @staticmethod
    def _round_half_up(n: float, decimals: int) -> float:
        """ Realizes the XCalibur type of rounding for m/z values.

        FreeStyle doesn't use it for abundance (I) values.
        """
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier


if __name__ == '__main__':

    path = 'C:/Users/skype/Desktop/ms-stuff/Fulvagra_ESI.raw'
    xrf = PyMSFileReader(path)
    data = xrf.get_averaged_spectrum_list_from_RT(90, 100)
    data = data['100.0-800.0__0']
