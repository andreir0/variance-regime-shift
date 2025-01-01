# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy"
# ]
# ///
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shift_in_variance import ShiftInVariance  # Assuming shift_in_variance.py is in the same directory

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Detect shifts in variance in a time series from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the time series data.")
    parser.add_argument("--time_col", type=str, default=None, help="Name of the column containing time data. If not provided, will use the first column.")
    parser.add_argument("--data_col", type=str, default=None, help="Name of the column containing the time series data. If not provided, will use the second column.")
    parser.add_argument("--significance_level", type=float, default=0.05, help="Significance level for the F-test.")
    parser.add_argument("--cutoff_length", type=int, default=20, help="Minimum length of a regime.")
    parser.add_argument("--huber_param", type=float, default=1.345, help="Tuning constant for Huber weighting.")
    parser.add_argument("--points_before_end", type=int, default=5, help="Number of points before the end of the time series to stop the test.")
    args = parser.parse_args()

    # Read data from CSV
    #todo: add error handling for reading csv file
    try:
        print(f"Attempting to read CSV file: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
        print(f"CSV file read successfully. DataFrame shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Extract time and data columns
    if args.time_col:
        try:
            print(f"Attempting to extract time column: {args.time_col}")
            time = df[args.time_col].to_numpy()
            print(f"Time column extracted successfully. Shape: {time.shape}")
        except KeyError:
            print(f"Error: Time column '{args.time_col}' not found in CSV.")
            return
    else:
        print("No time column specified, using the first column.")
        time = df.iloc[:, 0].to_numpy()
        print(f"Using the first column as time. Shape: {time.shape}")

    if args.data_col:
        try:
            print(f"Attempting to extract data column: {args.data_col}")
            data = df[args.data_col].to_numpy()
            print(f"Data column extracted successfully. Shape: {data.shape}")
        except KeyError:
            print(f"Error: Data column '{args.data_col}' not found in CSV.")
            return
    else:
        print("No data column specified, using the second column.")
        data = df.iloc[:, 1].to_numpy()
        print(f"Using the second column as data. Shape: {data.shape}")
    
    # # Plot ingested data
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, data, label='Ingested Data', color='blue')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title(f'Ingested Data from {args.csv_file}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Initialize ShiftInVariance object
    print("Initializing ShiftInVariance object.")
    shift_detector = ShiftInVariance(
        args.significance_level,
        args.cutoff_length,
        args.huber_param,
        args.points_before_end
    )

    # Detect shifts
    print("Detecting shifts in variance.")
    css, unweighted_variances, weighted_variances, regime_lengths, p_values, outliers, residuals = shift_detector.detect_shifts(data)
    print("Shift detection complete.")

    # Plot results
    title = f"Shifts in Variance for {args.csv_file}"
    print("Plotting shift detection results.")
    shift_detector.plot_results(time, data, css, unweighted_variances, weighted_variances, regime_lengths, p_values, outliers, residuals, title)
    print("Plotting complete.")

if __name__ == "__main__":
    main()