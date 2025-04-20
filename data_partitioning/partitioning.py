from itertools import product, combinations
from pathlib import Path
import json

import xarray as xr
import numpy as np
from sklearn import model_selection as ms
import pandas as pd
import matplotlib.pyplot as plt


class UniformTimeSeriesSplit:
    """Create train/test splits that are uniform throughout the year."""

    def __init__(
            self,
            n_splits: int = 4,
            test_size: float = None,
            year_fraction_interval: float = 1.0,
            gap: str = "5D",
            random_state: int = 1234,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = pd.Timedelta(gap)
        self.year_fraction_interval = year_fraction_interval
        self.random_state = random_state

    def split(self, X):
        X = pd.DatetimeIndex(X)

        if len(X) <= 1:
            raise ValueError("Time array must have at least 2 elements for splitting")

        interval = pd.Timedelta("365D") * self.year_fraction_interval

        # Calculate mean time difference with safety check
        time_diffs = np.diff(X)
        if len(time_diffs) == 0:
            print("Warning: Cannot calculate time differences. Using default value.")
            mean_diff = pd.Timedelta('1D')
        else:
            # Convert to timedeltas if they aren't already
            if not isinstance(time_diffs[0], pd.Timedelta):
                time_diffs = pd.TimedeltaIndex(time_diffs)

            # Use a reasonable default if mean is NaN
            mean_diff_ns = time_diffs.astype('timedelta64[ns]').mean()
            if pd.isna(mean_diff_ns):
                print("Warning: Mean time difference is NaN. Using default value.")
                mean_diff = pd.Timedelta('1D')
            else:
                mean_diff = pd.Timedelta(mean_diff_ns)

        gap_idx = max(1, int(self.gap / mean_diff))

        n_intervals_by_year = max(1, pd.Timedelta("365D") // interval)
        n_intervals = max(1, np.ceil((X[-1] - X[0]) / interval).astype("int"))
        n_test_intervals = max(1, np.floor(n_intervals * self.test_size).astype("int"))

        try:
            day_intervals = (
                    np.array(
                        [pd.Timedelta(f"{day}D") // interval for day in range(1, 367)],
                        dtype="int64",
                    )
                    + 1
            )
            map_day_to_interval = dict(zip(np.arange(1, 367, dtype="int64"), day_intervals))

            # Handle missing day_of_year values safely
            day_of_year = X.day_of_year.copy()
            if day_of_year.isna().any():
                print("Warning: Some dates have NaN day_of_year. Filling with 1.")
                day_of_year = day_of_year.fillna(1)

            element_interval_idx = (
                    day_of_year.map(map_day_to_interval).values
                    + (X.year - X.year.min()).values * n_intervals_by_year
            )
            interval_array = np.hstack(list(range(1, n_intervals_by_year + 1)) * 20)[
                             :n_intervals
                             ]

            # Ensure we don't have an empty combinations list
            if n_intervals <= n_test_intervals:
                print(f"Warning: n_intervals={n_intervals} <= n_test_intervals={n_test_intervals}. Adjusting.")
                n_test_intervals = max(1, n_intervals - 1)

            test_idx = np.array(
                list(combinations(range(n_intervals), n_test_intervals)), dtype="int"
            )

            if len(test_idx) == 0:
                print("Warning: No valid combinations found. Creating simple split.")
                # Create a simple split if no valid combinations
                mid_point = len(X) // 2
                train = np.ones(len(X), dtype=bool)
                train[mid_point:mid_point + len(X) // 4] = False
                test = ~train
                yield X[train], X[test]
                return

            rng = np.random.default_rng(self.random_state)
            rng.shuffle(test_idx)

            good_splits = 0
            for i, split in enumerate(test_idx):
                if good_splits == self.n_splits:
                    break

                if len(set(interval_array)) <= 1:
                    # Handle case with only one interval
                    is_good = True
                else:
                    is_good = set(interval_array[split]) == set(
                        range(1, n_intervals_by_year + 1)
                    )

                if not is_good:
                    continue

                train = ~np.isin(element_interval_idx, split + 1)

                if np.diff(train).size > 0:  # Check if train has more than one element
                    gaps = np.argwhere(np.diff(train) != 0) + np.arange(
                        -gap_idx / 2, gap_idx / 2
                    )
                    gaps[gaps < 0] = 0
                    gaps[gaps >= len(X)] = len(X) - 1
                    train[gaps.astype(int).ravel()] = False

                test = np.isin(element_interval_idx, split + 1)

                if np.diff(test).size > 0:  # Check if test has more than one element
                    gaps = np.argwhere(np.diff(test) != 0) + np.arange(
                        -gap_idx / 2, gap_idx / 2
                    )
                    gaps[gaps < 0] = 0
                    gaps[gaps >= len(X)] = len(X) - 1
                    test[gaps.astype(int).ravel()] = False

                # Ensure we have some data in both train and test
                if sum(train) > 0 and sum(test) > 0:
                    good_splits += 1
                    yield X[train], X[test]

            # If we didn't generate enough splits, create simple ones
            if good_splits < self.n_splits:
                print(f"Warning: Only generated {good_splits}/{self.n_splits} splits. Creating simple splits.")

                remaining = self.n_splits - good_splits
                chunk_size = len(X) // (remaining + 1)

                for i in range(remaining):
                    train = np.ones(len(X), dtype=bool)
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(X))
                    train[start_idx:end_idx] = False
                    test = ~train
                    yield X[train], X[test]

        except Exception as e:
            print(f"Error in split method: {e}")
            # Create a simple split as fallback
            mid_point = len(X) // 2
            train = np.ones(len(X), dtype=bool)
            train[mid_point:mid_point + len(X) // 4] = False
            test = ~train
            yield X[train], X[test]


def split_time(
        time_array: xr.DataArray,
        p: list,
        n_splits: int,
        shuffle: bool = False,
        seed: int = None,
        gap="5D",
):
    """Split time data into train, validation, and test sets."""
    print(f"Time array shape: {time_array.shape}")
    print(f"Time array range: {time_array.min().values} to {time_array.max().values}")

    if len(time_array) <= 1:
        raise ValueError("Time array must have at least 2 elements")

    # Ensure time_array is sorted
    if hasattr(time_array, 'sortby'):
        time_array = time_array.sortby(time_array)

    # Calculate time differences with safety checks
    try:
        time_diffs = np.diff(time_array)
        if len(time_diffs) == 0 or np.isnan(time_diffs).all() or np.all(time_diffs == pd.NaT):
            print("Warning: Cannot calculate mean time difference. Using default value of 1 day.")
            mean_diff = pd.Timedelta('1D')
        else:
            # Filter out NaN values if any
            valid_diffs = time_diffs[~(np.isnan(time_diffs) | (time_diffs == pd.NaT))]
            if len(valid_diffs) == 0:
                print("Warning: All time differences are invalid. Using default value of 1 day.")
                mean_diff = pd.Timedelta('1D')
            else:
                # Convert to a single representative value in nanoseconds
                if isinstance(valid_diffs[0], pd.Timestamp):
                    # Handle timestamp differences
                    mean_diff_ns = pd.TimedeltaIndex(valid_diffs).astype('timedelta64[ns]').mean()
                else:
                    # Handle direct timedeltas
                    mean_diff_ns = valid_diffs.astype('timedelta64[ns]').mean()

                if pd.isna(mean_diff_ns):
                    print("Warning: Mean time difference is NaN. Using default value of 1 day.")
                    mean_diff = pd.Timedelta('1D')
                else:
                    mean_diff = pd.Timedelta(mean_diff_ns)
    except Exception as e:
        print(f"Error calculating time differences: {e}")
        mean_diff = pd.Timedelta('1D')

    # Calculate gap index with safety
    gap_timedelta = pd.Timedelta(gap)
    gap_idx = max(1, int(gap_timedelta / mean_diff))
    print(f"Using gap_idx: {gap_idx} (equivalent to {gap})")

    # Split into train_val and test
    try:
        train_val, test = ms.train_test_split(
            time_array.values,
            test_size=p[-1],
            shuffle=shuffle,
            random_state=seed,
        )
    except Exception as e:
        print(f"Error in train_test_split: {e}")
        # Create simple split as fallback
        split_idx = int(len(time_array) * (1 - p[-1]))
        train_val = time_array.values[:split_idx]
        test = time_array.values[split_idx:]

    # Ensure train_val has enough elements
    if len(train_val) <= gap_idx:
        print(f"Warning: train_val set ({len(train_val)}) is smaller than gap_idx ({gap_idx}). Adjusting.")
        gap_idx = max(1, len(train_val) // 2)

    train_val = train_val[:-gap_idx]

    # Setup cross-validation
    cv_kwargs = {"gap": gap, "random_state": seed}
    cv = UniformTimeSeriesSplit(n_splits, test_size=p[1] / (1 - p[-1]), **cv_kwargs)
    split_kwargs = {"X": train_val}

    splits = []
    try:
        for train, val in cv.split(**split_kwargs):
            split = {
                "train": np.sort(train).astype(str).tolist(),
                "val": np.sort(val).astype(str).tolist(),
                "test": np.sort(test).astype(str).tolist(),
            }
            splits.append(split)
    except Exception as e:
        print(f"Error during CV splitting: {e}")
        # Create a simple split as fallback
        split_idx = int(len(train_val) * (1 - p[1] / (1 - p[-1])))

        for i in range(n_splits):
            # Create slightly different splits by rotating
            offset = (i * len(train_val) // n_splits) % len(train_val)
            rotated = np.roll(train_val, offset)

            train = rotated[:split_idx]
            val = rotated[split_idx:]

            split = {
                "train": np.sort(train).astype(str).tolist(),
                "val": np.sort(val).astype(str).tolist(),
                "test": np.sort(test).astype(str).tolist(),
            }
            splits.append(split)

    return splits


def plot_time_splits(time_array, splits, fn=None):
    """Plot the time splits for visualization."""
    time_array = time_array.values.astype("str").tolist()

    if len(splits) == 0:
        print("No splits to plot")
        return None

    fig, axs = plt.subplots(len(splits), sharex=True, figsize=(10, 4))

    # Handle single split case
    if len(splits) == 1:
        axs = [axs]

    for i, split in enumerate(splits):
        try:
            non_gaps = set(split["train"]) | set(split["val"]) | set(split["test"])
            gaps = list(set(time_array) - non_gaps)

            for set_ in ["train", "val", "test"]:
                if set_ in split and len(split[set_]) > 0:
                    date_index = pd.DatetimeIndex(split[set_])
                    axs[i].scatter(
                        date_index,
                        [0.5] * len(date_index),
                        marker="_",
                        linewidth=10,
                        label=set_,
                    )

            if len(gaps) > 0:
                try:
                    gap_index = pd.DatetimeIndex(gaps)
                    axs[i].scatter(
                        gap_index,
                        [0.5] * len(gap_index),
                        marker="_",
                        linewidth=10,
                        label="gap",
                    )
                except Exception as e:
                    print(f"Error plotting gaps: {e}")

            axs[i].set(yticks=[], yticklabels=[], ylim=(0.2, 0.8), ylabel=f"Split {i}")
        except Exception as e:
            print(f"Error plotting split {i}: {e}")
            continue

    # Add legend to the first subplot
    if len(axs) > 0:
        try:
            axs[0].legend(
                bbox_to_anchor=(0.0, 1.15, 1.0, 0.8),
                loc="lower left",
                ncol=4,
                mode="expand",
                borderaxespad=0.0,
                frameon=False,
            )
        except Exception as e:
            print(f"Error adding legend: {e}")

    plt.tight_layout()

    if fn is not None:
        try:
            plt.savefig(fn)
            print(f"Plot saved to {fn}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    return axs


def create_data_partitions(features_path, output_dir=None, seed=1):
    """Create data partitions for time series forecasting."""
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load features dataset
        print(f"Loading features from {features_path}")
        features_ds = xr.open_zarr(features_path)

        # Print dataset info for debugging
        print(f"Dataset dimensions: {features_ds.dims}")
        print(f"Dataset coordinates: {list(features_ds.coords)}")

        # Extract time and station arrays
        if 'forecast_reference_time' not in features_ds.coords:
            raise ValueError("Dataset does not contain 'forecast_reference_time' coordinate")

        # Find the actual time range in the data
        min_time = features_ds.forecast_reference_time.min().values
        max_time = features_ds.forecast_reference_time.max().values
        print(f"Available time range: {min_time} to {max_time}")

        # Select time range, adjust if needed
        try:
            time_slice = features_ds.sel(forecast_reference_time=slice("2017-01-01", "2022-01-01"))
            if len(time_slice.forecast_reference_time) <= 1:
                print("Warning: Not enough time points in selected range. Using full range.")
                time_array = features_ds.forecast_reference_time
            else:
                time_array = time_slice.forecast_reference_time
        except Exception as e:
            print(f"Error selecting time slice: {e}")
            print("Falling back to full time array")
            time_array = features_ds.forecast_reference_time

        print(f"Selected time array shape: {time_array.shape}")
        if hasattr(time_array, 'values') and len(time_array.values) > 0:
            print(f"Time range: {time_array.values.min()} to {time_array.values.max()}")

        station_array = features_ds.station
        print(f"Station array shape: {station_array.shape}")

        # Configuration for partitioning
        dp_config = {
            "p": [0.6, 0.2, 0.2],  # Train, val, test proportions
            "n_splits": 4,  # Number of CV splits
        }

        # Create time splits
        try:
            time_splits = split_time(
                time_array, **dp_config, seed=seed
            )
            print(f"Created {len(time_splits)} time splits")
        except Exception as e:
            print(f"Error creating time splits: {e}")
            return []

        # Plot splits if output directory is provided
        if output_dir and len(time_splits) > 0:
            try:
                plot_time_splits(time_array, time_splits, output_dir / "time_split.png")
            except Exception as e:
                print(f"Error plotting time splits: {e}")

        # Create final splits
        station_splits = [None]  # No station-specific splits
        all_splits = []

        for i, (t, s) in enumerate(product(time_splits, station_splits)):
            try:
                split = {}
                for set_name in ["train", "val", "test"]:
                    if s is None:
                        split[set_name] = {"forecast_reference_time": t[set_name]}
                    else:
                        split[set_name] = {"forecast_reference_time": t[set_name], "station": s[set_name]}

                if output_dir:
                    fn = output_dir / f"split_{i}.json"
                    with open(fn, "w") as f:
                        json.dump(split, f, indent=4)
                    print(f"Saved split {i} to {fn}")

                all_splits.append(split)
            except Exception as e:
                print(f"Error processing split {i}: {e}")

        print(f"Created {len(all_splits)} total splits")
        return all_splits

    except Exception as e:
        print(f"Error in create_data_partitions: {e}")
        import traceback
        traceback.print_exc()
        return []


# Example usage
if __name__ == "__main__":
    features_path = "data/features.zarr"
    output_dir = "results/data_partition"
    seed = 1

    splits = create_data_partitions(features_path, output_dir, seed=seed)

    print(f"Created {len(splits)} splits")
    for i, split in enumerate(splits):
        train_size = len(split["train"]["forecast_reference_time"])
        val_size = len(split["val"]["forecast_reference_time"])
        test_size = len(split["test"]["forecast_reference_time"])
        print(f"Split {i}: Train={train_size}, Val={val_size}, Test={test_size}")