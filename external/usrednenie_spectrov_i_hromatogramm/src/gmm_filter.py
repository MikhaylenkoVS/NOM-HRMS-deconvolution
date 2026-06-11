"""

This module contains the GMMNoiseFilter class, which is an
implementation of the proposed noise filtering algorithm.

"""
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM


# Globals ------------------------------------------------------------ #
S_FONT_SIZE = 8

C_HIST_ALPHA     = 0.3
C_HIST_BAR_WIDTH = 0.1

C_DIFF_THRESHOLD_MAJOR = 0.1
C_DIFF_THRESHOLD_MINOR = 0.04
C_CHECK_RADIUS         = 5

LOG = np.log10
EXP = lambda x: 10 ** x


# Matplotlib settings ------------------------------------------------ #
matplotlib.use('Agg')

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),
})

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams.update({'font.size': S_FONT_SIZE})


# Classes ------------------------------------------------------------ #
class GMMNoiseFilter:
    """ Implements the proposed method of noise filtering.

    Arguments
    ---------
    input_file_path : str
        Path to the input peak list in CSV format.
    selection_mode : int, default=0
        This attribute specifies how to select the number of
        gaussians.

        * 0 - find the bend in the BIC plot.
        * 1 - find the minumum point on the BIC plot.

    Methods
    -------
    save_denoised_data(folder_path: str) -> None
        Saves the peak list after noise filtering.
    save_noise_level_plot(folder_path: str) -> None:
        Saves a spectrum with the displayed noise level.
    save_bic_plot(folder_path: str) -> None:
        Saves the BIC plot.
    save_histogram_plot(folder_path: str) -> None:
        Saves the plot specific for this filtering method.
    """

    def __init__(self,
                 input_file_name: str,
                 selection_mode: int,
                 data: np.ndarray):

        self._input_file_name = input_file_name
        self._selection_mode = selection_mode

        self._input_data = None
        self._denoised_data = None

        self._bics = None
        self._noise_level = 0

        # Data for the histogram plot
        self._optimal_n_clust = None
        self._intersect_x = None
        self._x_min = None
        self._x_max = None
        self._x_axis = None
        self._y_axes = None
        self._gaussian_sum = None

        self._means = None
        self._covs = None
        self._weights = None

        # Process the input data ------------------------------------- #
        self._file_name = os.path.splitext(self._input_file_name)[0]

        self._input_data = data

        if selection_mode == 0:
            self._generate_bic_values(15)
            self._find_bic_bend()
        elif selection_mode == 1:
            self._generate_bic_values(10)
            self._find_bic_optimal_point()

        self._remove_noise()

    # ---------------------------------------------------------------- #
    # PUBLIC METHODS
    # ---------------------------------------------------------------- #
    def save_denoised_data(self, folder_path: str) -> None:
        """ Saves the peak list after noise filtering.

        Arguments
        ---------
        folder_path : str
            Path to the folder where the CSV file will be saved.
        """

        file_path = os.path.join(folder_path, self._file_name + '.csv')
        save_csv_file(file_path, self._denoised_data)

    def save_noise_level_plot(self, folder_path: str) -> None:
        """ Shows a spectrum with the displayed noise level.

        Arguments
        ---------
        folder_path : str
            Path to the folder where the PNG file will be saved.
        """

        delta_num_peaks = self._input_data.shape[0] -\
                          self._denoised_data.shape[0]
        delta_np_percent = (delta_num_peaks / self._input_data.shape[0]) * 100

        plt.figure(figsize=(6, 3), dpi=300)

        _, stems, _ = plt.stem(self._input_data[:, 0],
                               self._input_data[:, 1],
                               markerfmt=' ')
        plt.setp(stems, 'linewidth', 0.1, color='black', zorder=2)

        plt.hlines(y=self._noise_level,
                   xmin=self._input_data[:, 0][0],
                   xmax=self._input_data[:, 0][-1],
                   zorder=1,
                   linewidth=2,
                   color='magenta',
                   label='Noise threshold')

        # plt.xlim(left=200, right=400)
        plt.ylim(0, self._noise_level * 5)

        plt.title(f'Spectrum with noise level\n'\
                  f'{self._file_name}\n'\
                  f'Noise threshold = {self._noise_level:.3f}; '\
                  f'Cut {delta_num_peaks} peaks (-{delta_np_percent:.1f}%)')
        plt.xlabel('m/z', fontsize=S_FONT_SIZE)
        plt.ylabel('Intensity', fontsize=S_FONT_SIZE)

        plt.tight_layout(pad=0.5)

        image_name = self._file_name + '-level.png'
        image_path = os.path.join(folder_path, image_name)
        plt.savefig(image_path)

        plt.cla()
        plt.clf()
        plt.close()

    def save_bic_plot(self, folder_path: str) -> None:
        """ Saves the BIC plot.

        Arguments
        ---------
        folder_path : str
            Path to the folder where the PNG file will be saved.
        """

        x = np.arange(len(self._bics)) + 1

        plt.figure(figsize=(4, 4), dpi=300)

        plt.plot(x,
                 self._bics,
                 color='black',
                 marker='o',
                 linewidth=2,
                 markersize=3)
        
        plt.plot([self._optimal_n_clust],
                 [self._bics[self._optimal_n_clust - 1]],
                 color='black',
                 marker='d',
                 linewidth=3,
                 markersize=8)
        
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

        plt.title('BIC values for different numbers of Gaussians\n' +
                  f'{self._file_name}\n' + 
                  f'Selected number: {self._optimal_n_clust}')
        plt.xlabel('N(Gaussians)', fontsize=S_FONT_SIZE)
        plt.ylabel('BIC', fontsize=S_FONT_SIZE)
        plt.tight_layout(pad=0.5)

        image_name = self._file_name + '-bic.png'
        image_path = os.path.join(folder_path, image_name)
        plt.savefig(image_path)

        plt.cla()
        plt.clf()
        plt.close()

    def save_histogram_plot(self, folder_path: str) -> None:
        """ Saves the plot specific for this filtering method.

        Arguments
        ---------
        folder_path : str
            Path to the folder where the PNG file will be saved.
        """

        plt.figure(figsize=(4, 4), dpi=300)

        self._gen_hist_plot()

        plt.legend()
        plt.tight_layout(pad=0.5)

        image_name = self._file_name + '-hist.png'
        image_path = os.path.join(folder_path, image_name)
        plt.savefig(image_path)

        plt.cla()
        plt.clf()
        plt.close()

    # ---------------------------------------------------------------------- #
    # PRIVATE METHODS
    # ---------------------------------------------------------------------- #
    def _remove_noise(self) -> None:
        """ Detects and eliminates noise from a specified peak list. """

        self._noise_level = self._get_intensity_threshold()
        self._denoised_data = self._input_data[
                self._input_data[:, 1] > self._noise_level]

    def _get_intensity_threshold(self) -> float:
        """ Calculates the noise intensity level from data. """

        x = LOG(self._input_data[:, 1, None])

        # For the histogram plot
        self._x_min  = np.min(x)
        self._x_max  = np.max(x)
        self._x_axis = np.linspace(self._x_min, self._x_max, 200)

        gmm = GMM(n_components=self._optimal_n_clust,
                  max_iter=1000,
                  random_state=135,
                  covariance_type='full')

        # get the gaussian parameters
        means   = gmm.fit(x).means_
        covs    = gmm.fit(x).covariances_
        weights = gmm.fit(x).weights_

        self._means   = means
        self._covs    = covs
        self._weights = weights

        # For the histogram plot
        self._y_axes = []

        for i in range(means.shape[0]):

            y = norm.pdf(self._x_axis,
                         float(means[i][0]),
                         np.sqrt(float(covs[i][0][0])))
            y *= weights[i]
            self._y_axes.append(y)

        self._gaussian_sum = np.sum(self._y_axes, axis=0)[:, None]

        heights = []

        for _ in range(means.shape[0]):
            heights.append(np.max(self._y_axes[i]))

        # find the first and second gaussians
        indices    = means[:, 0].argsort()
        first_idx  = indices[0]
        second_idx = indices[1]

        # get the parameters of the first and second gaussians
        mu1 = float(means[first_idx][0])
        mu2 = float(means[second_idx][0])

        sigm1 = float(covs[first_idx][0][0])
        sigm2 = float(covs[second_idx][0][0])

        w1 = float(weights[first_idx])
        w2 = float(weights[second_idx])

        # find the intersection points
        self._intersect_x = self._get_intersection_x(
                mu1, mu2, np.sqrt(sigm1), np.sqrt(sigm2), w1, w2)

        return EXP(self._intersect_x)

    def _get_intersection_x(self, m1: float, m2: float, std1: float,
            std2: float, w1: float, w2: float) -> float:
        """ Calculates the x of the intersection point of 2 gaussians. """

        A = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
        B = m2 / (std2 ** 2) - m1 / (std1 ** 2)
        C = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) -\
                np.log(std2/ std1) - np.log(w1 / w2)

        roots = np.roots([A, B, C])

        if np.min(roots) > m1:
            result = np.min(roots)
        else:
            result = np.max(roots)

        return result
    
    def _generate_bic_values(self, max_clusters: int) -> None:
        """ Calculates the BIC values.

        The number of gaussians in the range of 1 to 'max_clusters'
        is used.

        Parameters
        ----------
        max_clusters : int
            Maximum number of gaussians to test.
        """

        self._bics = []
        x = LOG(self._input_data[:, 1, None])

        for i in range(1, max_clusters + 1):

            gmm = GMM(n_components=i,
                      max_iter=1000,
                      random_state=42,
                      covariance_type='full')

            gmm.fit(x).predict(x)
            self._bics.append(gmm.bic(x))

    def _find_bic_optimal_point(self) -> int:
        """ Finds the optimal number of cluster as the minimum point. """

        self._optimal_n_clust = np.argmin(self._bics) + 1

    def _find_bic_bend(self) -> int:
        """ Finds the boundary of a steep slope on the BIC plot. """

        diff = []
        bend_idx = 0
        bic_grad_inv = np.flip(np.gradient(self._bics))
        bic_grad_inv /= abs(np.min(bic_grad_inv))

        for i in range(2, len(bic_grad_inv)):
            last_mean = np.mean(bic_grad_inv[:i])
            diff.append((len(bic_grad_inv) - i, abs(bic_grad_inv[i] - last_mean)))

        for k in range(len(diff)):
            if diff[k][1] > C_DIFF_THRESHOLD_MAJOR:
                bend_idx = k
                self._optimal_n_clust = diff[k][0]
                break

        for m in range(1, C_CHECK_RADIUS + 1):
            if diff[bend_idx - m][1] > C_DIFF_THRESHOLD_MINOR:
                self._optimal_n_clust = diff[bend_idx - m][0]
            else:
                break

    def _gen_hist_plot(self) -> None:
        """ Renders additional plot for this filtering method. """

        plt.hist(LOG(self._input_data[:, 1, None]),
                 density=True,
                 color='black',
                 alpha=C_HIST_ALPHA,
                 bins=np.arange(self._x_min, self._x_max, C_HIST_BAR_WIDTH))

        # Sort the plot legend
        indices = self._means[:, 0].argsort()

        for i in indices:
            plt.plot(self._x_axis, self._y_axes[i], lw=2, c=f'C{i}',
                     label=f'mu: {float(self._means[i][0]):.2f}; ' +\
                           f'sigma: {float(np.sqrt(self._covs[i][0][0])):.2f}; ' +\
                           f'w: {float(self._weights[i]):.2f}')

        plt.plot(self._x_axis, self._gaussian_sum, lw=2, c=f'C{i + 1}',
                 ls='dashed', label='Sum of gaussins', zorder=-1)
        plt.vlines(x=self._intersect_x, ymin=0,
                   ymax=np.max(self._gaussian_sum), lw=2, color='magenta',
                   zorder=50, label='Noise threshold')

        plt.text(x=self._intersect_x + 0.1, y=np.max(self._gaussian_sum) / 2,
                 s=f'{self._intersect_x:.3f}', fontsize=S_FONT_SIZE,
                 clip_on=True, color='magenta', backgroundcolor='white')

        plt.title(f'GMM noise threshold. '\
                  f'Number of Gaussians: {self._optimal_n_clust}\n'
                  f'{self._file_name}\n'\
                  f'log10(noise threshold) = {self._intersect_x:.3f}')

        plt.xlabel('log10(Intensity)', fontsize=S_FONT_SIZE)
        plt.ylabel('density', fontsize=S_FONT_SIZE)

        plt.gca().tick_params(axis='both', labelsize=S_FONT_SIZE)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))


if __name__ == '__main__':

    gmm = GMMNoiseFilter('data/test.csv', selection_mode=0)
    gmm.save_bic_plot('.')
    gmm.save_noise_level_plot('.')
    gmm.save_histogram_plot('.')
    gmm.save_denoised_data('.')
