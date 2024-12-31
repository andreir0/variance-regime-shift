# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib"
# ]
# ///
import numpy as np
import pandas as pd
from scipy.stats import f
from typing import List, Tuple
import matplotlib.pyplot as plt

class ShiftInVariance:
    """
    Detects shifts in variance in a time series using a sequential approach.
    """

    def __init__(self, significance_level: float, cutoff_length: int, huber_param: float, points_before_end: int):
        """
        Initializes the ShiftInVariance object.

        Args:
            significance_level (float): Significance level for the F-test.
            cutoff_length (int): Minimum length of a regime.
            huber_param (float): Tuning constant for Huber weighting.
            points_before_end (int): Number of points before the end of the time series to stop the test.
        """
        self.prob = significance_level
        self.cutoff = cutoff_length
        self.sngHuber = huber_param
        self.intPointsBeforeEnd = points_before_end

    def _huber_weighted_var(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates the weighted variance using Huber weighting.

        Args:
            x (np.ndarray): Time series data for the current regime.

        Returns:
            Tuple[float, float, float]: Weighted variance, sum of weights, and sum of squared weights.
        """
        n = len(x)
        x_abs = np.abs(x)
        scale = 1.4826 * np.median(x_abs)
        
        for _ in range(2):
            sum_of_weights = 0.0
            sum_of_weights2 = 0.0
            weighted_var = 0.0
            for i in range(n):
                if x_abs[i] == 0:
                    x_weight = 1.0
                else:
                    x_weight = min(1, self.sngHuber * scale / x_abs[i])
                    x_weight = x_weight * x_weight
                sum_of_weights += x_weight
                sum_of_weights2 += x_weight * x_weight
                weighted_var += x[i] * x[i] * x_weight
            
            x_weight_coef = sum_of_weights - (sum_of_weights2 / sum_of_weights)
            weighted_var = weighted_var / x_weight_coef
            scale = np.sqrt(weighted_var)
        
        return weighted_var, sum_of_weights, sum_of_weights2

    def _ss_up(self, x: np.ndarray, current_year: int, var2_up: float) -> float:
        """
        Calculates the sum of squares when var2 > var1.

        Args:
            x (np.ndarray): Time series data.
            current_year (int): Current year index.
            var2_up (float): Upper threshold for variance.

        Returns:
            float: Sum of squares.
        """
        ss_up = 0.0
        sum_of_weights = 0.0
        scale = np.sqrt(var2_up)
        n = len(x)
        for i in range(current_year, min(current_year + self.cutoff, n)):
            if x[i] == 0:
                x_weight = 1.0
            else:
                x_weight = min(1, self.sngHuber * scale / abs(x[i]))
                x_weight = x_weight * x_weight
            ss_up += (x[i] * x[i] * x_weight - var2_up)
            sum_of_weights += x_weight
            if ss_up < 0:
                return 0.0
        
        if sum_of_weights > 0:
            return ss_up / sum_of_weights
        return 0.0

    def _ss_down(self, x: np.ndarray, current_year: int, var2_down: float) -> float:
        """
        Calculates the sum of squares when var2 < var1.

        Args:
            x (np.ndarray): Time series data.
            current_year (int): Current year index.
            var2_down (float): Lower threshold for variance.

        Returns:
            float: Sum of squares.
        """
        ss_down = 0.0
        sum_of_weights = 0.0
        h = 1.5 if self.sngHuber < 1.5 else self.sngHuber
        scale = np.sqrt(var2_down)
        n = len(x)

        for i in range(current_year, min(current_year + self.cutoff, n)):
            if x[i] == 0:
                x_weight = 1.0
            else:
                x_weight = min(1, h * scale / abs(x[i]))
                x_weight = x_weight * x_weight
            ss_down += (x[i] * x[i] * x_weight - var2_down)
            sum_of_weights += x_weight
            if ss_down > 0:
                return 0.0
        
        if sum_of_weights > 0:
            return ss_down / sum_of_weights
        return 0.0

    def detect_shifts(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects shifts in variance in the given time series.

        Args:
            data (np.ndarray): Time series data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Cumulative sum of squares (CSS)
                - Unweighted variance for each regime
                - Weighted variance for each regime
                - Length of each regime
                - P-values for each shift
        """
        n = len(data)
        css = np.zeros(n)
        unweighted_variances = np.zeros(n)
        weighted_variances = np.zeros(n)
        regime_lengths = np.zeros(n, dtype=int)
        p_values = np.zeros(n)
        outliers = np.array([''] * n, dtype=object)
        residuals = np.zeros(n)

        # Initial regime
        weighted_var, sum_of_weights, sum_of_weights2 = self._huber_weighted_var(data[:self.cutoff])
        var2_up = weighted_var * f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
        var2_down = weighted_var / f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
        change_point = 0
        number_of_regimes = 0
        regime_start = 0

        for current_year in range(1, n):
            if data[current_year] * data[current_year] > var2_up:
                css[current_year] = self._ss_up(data, current_year, var2_up)
            elif data[current_year] * data[current_year] < var2_down:
                css[current_year] = self._ss_down(data, current_year, var2_down)
            else:
                css[current_year] = 0

            if css[current_year] != 0 and current_year > (n - self.cutoff):
                if current_year > n - self.intPointsBeforeEnd:
                    css[current_year] = 0
                break

            if css[current_year] == 0:
                if current_year >= change_point + self.cutoff:
                    if data[current_year] == 0:
                        x_weight = 1
                    else:
                        x_weight = min(1, self.sngHuber * np.sqrt(weighted_var) / abs(data[current_year]))
                        x_weight = x_weight * x_weight
                    x_weight_coef = sum_of_weights - (sum_of_weights2 / sum_of_weights)
                    weighted_var = weighted_var * x_weight_coef + x_weight * data[current_year] * data[current_year]
                    sum_of_weights += x_weight
                    sum_of_weights2 += x_weight * x_weight
                    x_weight_coef = sum_of_weights - (sum_of_weights2 / sum_of_weights)
                    weighted_var = weighted_var / x_weight_coef
                    var2_up = weighted_var * f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
                    var2_down = weighted_var / f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
            else:
                change_point = current_year
                weighted_var, sum_of_weights, sum_of_weights2 = self._huber_weighted_var(data[change_point:change_point + self.cutoff])
                var2_up = weighted_var * f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
                var2_down = weighted_var / f.ppf(1 - self.prob / 2, self.cutoff - 1, self.cutoff - 1)
        
        # Calculate regime characteristics
        for i in range(n):
            if abs(css[i]) > 0 or i == n - 1:
                number_of_regimes += 1
                regime_end = i - 1 if i < n - 1 else i
                regime_length = regime_end - regime_start + 1
                
                regime_data = data[regime_start:regime_end+1]
                unweighted_var = np.sum(regime_data * regime_data) / regime_length if regime_length > 1 else -999
                weighted_var_regime, _, _ = self._huber_weighted_var(regime_data) if regime_length > 1 else (-999, 0, 0)
                
                unweighted_variances[regime_start:regime_end+1] = unweighted_var
                weighted_variances[regime_start:regime_end+1] = weighted_var_regime
                regime_lengths[regime_start:regime_end+1] = regime_length
                
                if weighted_var_regime > 0:
                    scale = np.sqrt(weighted_var_regime)
                    for j in range(regime_start, regime_end + 1):
                        residuals[j] = data[j] / scale
                        if abs(residuals[j]) > self.sngHuber:
                            outliers[j] = str(self.sngHuber / abs(residuals[j]))
                else:
                    residuals[regime_start:regime_end+1] = data[regime_start:regime_end+1]
                
                regime_start = i
        
        # Calculate p-values
        for i in range(1, n - 1):
            if abs(css[i]) > 0:
                nf1 = regime_lengths[i] - 1
                nf2 = regime_lengths[i+1] - 1
                if nf1 < 1 or nf2 < 1:
                    p_values[i+1] = np.nan
                else:
                    f_ratio = weighted_variances[i+1] / weighted_variances[i]
                    if f_ratio > 1:
                        p_values[i+1] = f.cdf(1 / f_ratio, nf1, nf2) * 2
                    else:
                        p_values[i+1] = f.cdf(f_ratio, nf2, nf1) * 2
        
        return css, unweighted_variances, weighted_variances, regime_lengths, p_values, outliers, residuals
    
    def plot_results(self, time: np.ndarray, data: np.ndarray, css: np.ndarray, unweighted_variances: np.ndarray, weighted_variances: np.ndarray, regime_lengths: np.ndarray, p_values: np.ndarray, outliers: np.ndarray, residuals: np.ndarray, title: str):
        """
        Plots the results of the shift detection.

        Args:
            time (np.ndarray): Time axis data.
            data (np.ndarray): Original time series data.
            css (np.ndarray): Cumulative sum of squares.
            unweighted_variances (np.ndarray): Unweighted variances.
            weighted_variances (np.ndarray): Weighted variances.
            regime_lengths (np.ndarray): Lengths of regimes.
            p_values (np.ndarray): P-values for shifts.
            outliers (np.ndarray): Outlier weights.
            residuals (np.ndarray): Normalized residuals.
            title (str): Title of the plot.
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot original data
        axs[0].plot(time, data, label='Original Data', color='black')
        
        # Add vertical lines where CSS is not zero
        for i, val in enumerate(css):
            if val != 0:
                axs[0].axvline(time[i], color='red', linestyle='--', alpha=0.7, label='Shift Point' if i == 0 else "")
        
        axs[0].set_ylabel('Value')
        axs[0].set_title(title)
        axs[0].legend()
        axs[0].axhline(0, color='black', linestyle='--') # Added baseline
        
        # Plot CSS with baseline
        axs[1].bar(time, css, label='RSSI', color='green')
        axs[1].set_title(f'RSSI (p={self.prob}, cutoff={self.cutoff}, Huber={self.sngHuber})')
        axs[1].axhline(0, color='black', linestyle='--')  # Add baseline here
        axs[1].set_ylabel('RSSI')
        axs[1].legend()
        
        # Plot residuals
        axs[2].plot(time, residuals, label='Normalized Residuals', color='purple')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Residuals')
        axs[2].legend()
        axs[2].axhline(0, color='black', linestyle='--') # Added baseline
        
        plt.tight_layout()
        plt.show()