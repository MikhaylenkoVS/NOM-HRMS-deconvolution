#!/usr/bin/python3
""" The main module of the application. """
import colorama as clr
from ctypes import *
import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import numpy as np

import gmm_filter
import pymsfilereader as msfr


windll.shcore.SetProcessDpiAwareness(True)


# Globals ------------------------------------------------------------------- #
_C_CHECKED_APPLY_NOISE_FILTER = True
_C_CHECKED_SAVE_NOISE_LEVEL_PLOT = False
_C_CHECKED_SAVE_HISTOGRAM_PLOT = False


class Application(tk.Frame):
    """ Describes the GUI of the application. """

    def __init__(self, master: tk.Tk = None):
        super().__init__(master)

        master.title('ms-average-app')
        master.resizable(False, False)

        self._apply_noise_filter = tk.BooleanVar()
        self._save_noise_level_plot = tk.BooleanVar()
        self._save_histogram_plot = tk.BooleanVar()

        self._current_file_name = None
        self._plot_folder_path = ''
        self._start_time = None
        self._step_time = None
        self._end_time = None

        self._xrf = None
        self._min_rt = None
        self._max_rt = None

        self.pack()
        self._create_widgets()

        clr.init()

    # ----------------------------------------------------------------------- #
    # PRIVATE METHODS
    # ----------------------------------------------------------------------- #
    def _create_widgets(self) -> None:
        """ Creates the GUI. """

        # Open a file ---------------------------------------------------------
        raw_file_button = tk.Button(self, text='Open .RAW file',
                                  command=self._ask_raw_file, bg='#c6b4ed')
        self._raw_file_entry = ttk.Entry(self, state='readonly')

        raw_file_button.grid(row=0, column=0, sticky='nsew')
        self._raw_file_entry.grid(row=0, column=1, columnspan=2, sticky='nsew')

        # Ranges --------------------------------------------------------------
        start_time_label = tk.Label(self, text='Start time, min',
                                    justify=tk.CENTER)
        step_time_label = tk.Label(self, text='Time step, min',
                                   justify=tk.CENTER)
        end_time_label = tk.Label(self, text='End time, min',
                                  justify=tk.CENTER)

        self._start_time_entry = ttk.Entry(self, justify=tk.RIGHT)
        self._step_time_entry = ttk.Entry(self, justify=tk.RIGHT)
        self._end_time_entry = ttk.Entry(self, justify=tk.RIGHT)

        start_time_label.grid(row=1, column=0, sticky='nsew')
        step_time_label.grid(row=1, column=1, sticky='nsew')
        end_time_label.grid(row=1, column=2, sticky='nsew')

        self._start_time_entry.grid(row=2, column=0, sticky='nsew')
        self._step_time_entry.grid(row=2, column=1, sticky='nsew')
        self._end_time_entry.grid(row=2, column=2, sticky='nsew')

        # Output folder -------------------------------------------------------
        output_folder_button = tk.Button(
            self, text='Select the output folder',
            command=self._get_output_folder, bg='#c6b4ed')
        self._output_folder_entry = ttk.Entry(self, width=30, justify=tk.LEFT,
                                              state='readonly')

        output_folder_button.grid(row=3, column=0, sticky='nsew')
        self._output_folder_entry.grid(
            row=3, column=1, columnspan=2, sticky='nsew')

        # Noise removal -------------------------------------------------------
        apply_noise_filter_check = tk.Checkbutton(
            self, text='Apply GMM noise filter',
            variable=self._apply_noise_filter)
        save_noise_level_plot_check = tk.Checkbutton(
            self, text='Save noise level plots',
            variable=self._save_noise_level_plot)
        save_histogram_plot_check = tk.Checkbutton(
            self, text="Save histogram plots",
            variable=self._save_histogram_plot)

        apply_noise_filter_check.grid(row=5, column=0, sticky='nsew')
        save_noise_level_plot_check.grid(row=5, column=1, sticky='nsew')
        save_histogram_plot_check.grid(row=5, column=2, sticky='nsew')

        if _C_CHECKED_APPLY_NOISE_FILTER:
            apply_noise_filter_check.select()

        if _C_CHECKED_SAVE_NOISE_LEVEL_PLOT:
            save_noise_level_plot_check.select()

        if _C_CHECKED_SAVE_HISTOGRAM_PLOT:
            save_histogram_plot_check.select()

        # Run -----------------------------------------------------------------
        self.run_button = tk.Button(self, text='Run!', height=2,
                                    command=self._run, bg='#aaefa2')
        self.run_button.grid(row=6, column=0, columnspan=3, sticky='nsew')

    def _ask_raw_file(self) -> None:
        """ Asks a file path and saves it to the entry. """

        filename = filedialog.askopenfilename(
            initialdir='../',
            title='Open .RAW file',
            filetypes=(('mass spectra', '*.raw'), ('all files', '*.*')))

        self._raw_file_entry.configure(state=tk.ACTIVE)

        self._raw_file_entry.delete(0, tk.END)
        self._raw_file_entry.insert(0, filename)

        self._raw_file_entry.configure(state='readonly')

        self._raw_file_entry.update()
        self._raw_file_entry.xview_moveto(1)

    def _get_output_folder(self) -> None:
        """ Asks a folder path and saves it to the entry. """

        dirname = filedialog.askdirectory(
            initialdir='../',
            title='Select the output folder')

        self._output_folder_entry.configure(state=tk.ACTIVE)

        self._output_folder_entry.delete(0, tk.END)
        self._output_folder_entry.insert(0, dirname)

        self._output_folder_entry.configure(state='readonly')

        self._output_folder_entry.update()
        self._output_folder_entry.xview_moveto(1)

    def _run(self) -> None:
        """ Processes a specified RAW file. """

        if not self._raw_file_entry.get():
            return
        elif not self._output_folder_entry.get():
            return

        file_name = os.path.basename(self._raw_file_entry.get())
        self._current_file_name = os.path.splitext(file_name)[0]

        self._output_path = os.path.join(self._output_folder_entry.get(),
                                         self._current_file_name)

        if not os.path.exists(self._output_path):
            os.mkdir(self._output_path)

        self._plot_folder_path = os.path.join(
            self._output_path, 'plots')

        if ((self._save_noise_level_plot.get()
            or self._save_histogram_plot.get())
            and self._apply_noise_filter.get()
            and not os.path.exists(self._plot_folder_path)):

            os.mkdir(self._plot_folder_path)

        self._print_file_name_as_title()
        self._print_checkbox_values()

        self._load_raw_file(self._raw_file_entry.get())

        self._start_time = (
            float(self._start_time_entry.get())
            if self._start_time_entry.get()
            else -999.0
        )
        self._step_time = (
            float(self._step_time_entry.get())
            if self._step_time_entry.get()
            else -999.0
        )
        self._end_time = (
            float(self._end_time_entry.get())
            if self._end_time_entry.get()
            else -999.0
        )

        self._update_time_values()

        if self._step_time == 0:
            self._average_one_time_range(self._start_time, self._end_time)
        else:
            for st in np.arange(self._start_time,
                                self._end_time,
                                self._step_time):
                current_start_time = st
                current_end_time = st + self._step_time

                if current_end_time > self._end_time:
                    current_end_time = self._end_time
                
                self._average_one_time_range(
                        current_start_time, current_end_time)

        self._print_done()

    def _save_csv_file(self, file_path: str, data: np.ndarray) -> None:
        """ Saves data to a CSV file. """

        with open(file_path, 'w') as wf:
            wf.write('m/z,Intensity\n')
            for pair in data:
                wf.write(f'{self._round_half_up(pair[0], 12)},'
                         f'{self._round_half_up(pair[1], 12)}\n')

    def _save_tab_separated_file(self, file_path: str,
                                 data: np.ndarray,
                                 all_data: bool=False) -> None:
        """ Saves data to a tab separated TXT file. """

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

    def _save_peak_lists(self, spectrum_lists: dict,
                         start_time: float, end_time: float) -> None:
        """ """

        for list_name, spectrum_list in spectrum_lists.items():

            save_file_name = (
                f'[{self._current_file_name}]_'
                f'{start_time:.1f}-{end_time:.1f}__'
                f'{list_name}-avrg'\
                f'{"-dn" if self._apply_noise_filter.get() else ""}'
            )

            save_file_path = os.path.join(self._output_path,
                                          save_file_name)

            self._save_csv_file(save_file_path + '.csv', spectrum_list)
            self._save_tab_separated_file(save_file_path + '.txt',
                                          spectrum_list)
 
    def _round_half_up(self, n: float, decimals: int) -> float:
        """ Realizes the XCalibur type of rounding for m/z values.

        FreeStyle doesn't use it for abundance (I) values.
        """
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier

    def _print_file_name_as_title(self) -> None:
        """ Pretty prints the name of the processed RAW file. """

        if self._current_file_name:
            print(
                clr.Back.CYAN + clr.Style.BRIGHT,
                'RAW FILE:',
                self._current_file_name, 10 * '=',
                clr.Style.RESET_ALL,
                '\n'
            )

    def _print_checkbox_values(self) -> None:
        """ Pretty prints the values of the GUI checkboxes. """

        print('Apply GMM noise filter:', clr.Fore.MAGENTA,
              self._apply_noise_filter.get(), clr.Style.RESET_ALL)
        print('Save noise level plots:', clr.Fore.MAGENTA,
              self._save_noise_level_plot.get(), clr.Style.RESET_ALL)
        print('Save histogram plots:', clr.Fore.MAGENTA,
              self._save_histogram_plot.get(), clr.Style.RESET_ALL)

    def _print_done(self) -> None:
        """ Pretty prints the 'DONE' word. """

        print(
            clr.Fore.GREEN,
            '\r\n' + 10 * '=', 'DONE!', 10 * '=',
            clr.Style.RESET_ALL,
            end='\n'
        )

    def _load_raw_file(self, raw_file_path) -> None:
        """ Loads a ThermoRAW file. """

        self._xrf = msfr.PyMSFileReader(raw_file_path)

        self._min_rt = self._xrf.get_min_rt().value
        self._max_rt = self._xrf.get_max_rt().value

    def _update_time_values(self) -> None:
        """ Updates the input time values. """

        self._start_time = (
            self._start_time
            if self._start_time > self._min_rt
            else self._min_rt
        )
        self._step_time = (
            self._step_time
            if self._step_time > 0
            else 0
        )
        self._end_time = (
            self._end_time
            if self._end_time < self._max_rt
            else self._max_rt
        )

    def _average_one_time_range(
            self, start_time: float, end_time: float) -> None:
        """ """

        print(
            '\nAveraging mass spectra from',
            clr.Fore.CYAN,
            f'{start_time:.2f}',
            clr.Style.RESET_ALL,
            'to',
            clr.Fore.CYAN,
            f'{end_time:.2f}',
            clr.Style.RESET_ALL,
            'minutes...')

        spectrum_lists = self._xrf.get_averaged_spectrum_list_from_RT(
            start_rt=c_double(start_time), end_rt=c_double(end_time))

        key_name = f'{start_time:.1f}-{end_time:.1f}'

        if self._apply_noise_filter.get():
            for list_name, spectrum_list in spectrum_lists.items():

                print('Noise filtering...')

                input_file_name = (
                    f'[{self._current_file_name}]_'
                    f'{start_time:.1f}-{end_time:.1f}__'
                    f'{list_name}-avrg'\
                )

                gmm = gmm_filter.GMMNoiseFilter(
                    input_file_name=input_file_name,
                    selection_mode=0,
                    data=spectrum_list)

                spectrum_lists[list_name] = gmm._denoised_data

                print(
                    clr.Fore.YELLOW,
                    '\r' + f'{list_name}:',
                    clr.Style.RESET_ALL,
                    f'NUM PEAKS WAS',
                    clr.Fore.RED,
                    str(spectrum_list.shape[0]),
                    clr.Style.RESET_ALL,
                    f'NOW:',
                    clr.Fore.RED,
                    str(gmm._denoised_data.shape[0]),
                    clr.Style.RESET_ALL)

                if self._save_noise_level_plot.get():
                    gmm.save_noise_level_plot(self._plot_folder_path)

                if self._save_histogram_plot.get():
                    gmm.save_histogram_plot(self._plot_folder_path)

        self._save_peak_lists(spectrum_lists, start_time, end_time)


if __name__ == '__main__':

    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

