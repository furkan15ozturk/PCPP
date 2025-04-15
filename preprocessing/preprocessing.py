import polars as pl
import numpy as np
import xarray as xr
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def filter_stations(targets, threshold=0.9):
    """Only keep stations with completeness above a certain threshold, as in original code."""
    dims = ["forecast_reference_time", "t"]
    n_stations = len(targets.station.values)
    ds = targets.stack(s=dims).to_array("var")
    missing = np.isnan(ds).sum("var")
    completeness = (missing == 0).sum("s") / len(ds.s)
    targets = targets.where(completeness > threshold, drop=True)
    n_bad_stations = n_stations - len(targets.station.values)
    LOGGER.info(f"Filtered out {n_bad_stations} out of {n_stations}")
    return targets


def preprocess_cosmo_data(filepath, output_dir):
    """Process COSMO-E CSV data to match the format needed for the model"""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Reading data from {filepath}")

    # Read CSV with appropriate separator
    # Skip the first 4 lines which are metadata and parse units/member info
    df = pl.read_csv(
        filepath,
        separator=';',
        skip_rows=17,  # Skip all the metadata rows
        has_header=False
    )

    # Assign the header based on the first few lines in your data
    # Stripping out empty columns
    header = ['stn', 'time', 'leadtime'] + \
             ['T_2M_' + str(i).zfill(2) for i in range(21)] + \
             ['FF_10M_' + str(i).zfill(2) for i in range(21)] + \
             ['DD_10M_' + str(i).zfill(2) for i in range(21)] + \
             ['TOT_PREC_' + str(i).zfill(2) for i in range(21)] + \
             ['RELHUM_2M_' + str(i).zfill(2) for i in range(21)] + \
             ['DURSUN_' + str(i).zfill(2) for i in range(21)]

    # Rename columns
    df.columns = header

    # Drop rows where stn column is empty
    df = df.filter(pl.col('stn') != '')

    # Convert time and leadtime to appropriate formats
    df = df.with_columns([
        pl.col('time').str.to_datetime('%Y%m%d %H:%M'),
        pl.col('leadtime').str.split(':').list.first().cast(pl.Int32) * 60 +
        pl.col('leadtime').str.split(':').list.get(1).cast(pl.Int32)
    ])

    # Convert leadtime from minutes to hours and create timedelta
    df = df.with_columns([
        (pl.col('leadtime') / 60).alias('leadtime_hours')
    ])

    # Convert remaining columns to float type, handling missing values (-999.0)
    numeric_cols = [col for col in df.columns if col not in ['stn', 'time', 'leadtime']]
    for col in numeric_cols:
        df = df.with_columns([
            pl.when(pl.col(col) == "-999.0")
            .then(None)
            .otherwise(pl.col(col))
            .cast(pl.Float64)
            .alias(col)
        ])

    # Filter leadtimes between 3 and 24 hours as in the original code
    df = df.filter((pl.col('leadtime_hours') >= 3) & (pl.col('leadtime_hours') <= 24))

    # Filter for forecast time 0 hours as in the original code
    df = df.filter(pl.col('time').dt.hour() == 0)

    # Create features dataframe with ensemble averages
    LOGGER.info("Calculating ensemble averages")

    # Calculate ensemble averages for temperature and relative humidity
    t_cols = [f'T_2M_{i:02d}' for i in range(21)]
    rh_cols = [f'RELHUM_2M_{i:02d}' for i in range(21)]

    # Calculate ensemble averages
    features_df = df.select([
        'stn',
        'time',
        'leadtime_hours',
        pl.mean(t_cols).alias('coe:air_temperature_ensavg'),
        pl.mean(rh_cols).alias('coe:relative_humidity_ensavg')
    ])

    # Calculate dewpoint temperature for each ensemble member
    LOGGER.info("Calculating dew point temperature")

    # Function to calculate dewpoint from T and RH
    def calculate_dewpoint(t, rh):
        """Calculate dewpoint using Magnus formula"""
        a = 17.368
        b = 238.83
        gamma = np.log(rh / 100) + (a * t) / (b + t)
        return (b * gamma) / (a - gamma)

    # Calculate dewpoint for each member
    td_members = []
    for i in range(21):
        # Extract the data for calculation
        t_data = df.select(pl.col(f'T_2M_{i:02d}')).to_numpy().flatten()
        rh_data = df.select(pl.col(f'RELHUM_2M_{i:02d}')).to_numpy().flatten()

        # Calculate dewpoints
        dewpoint = calculate_dewpoint(t_data, rh_data)
        td_members.append(dewpoint)

    # Calculate ensemble average of dewpoint temperature
    td_avg = np.nanmean(np.vstack(td_members), axis=0)

    # Add calculated variables to the features dataframe
    features_df = features_df.with_columns([
        pl.Series('coe:dew_point_temperature_ensavg', td_avg),
        (pl.col('coe:air_temperature_ensavg') - pl.col('coe:dew_point_temperature_ensavg')).alias(
            'coe:dew_point_depression_ensavg')
    ])

    # Get station coordinates and elevation from the metadata
    # Extract from the grid_idx, grid_lat, grid_lon, and grid_height from the metadata
    # For this example, read the metadata manually
    with open(filepath, 'r') as f:
        lines = f.readlines()

    grid_idx_line = next((line for line in lines if 'Grid_idx:' in line), None)
    grid_lat_line = next((line for line in lines if 'Grid_latitude:' in line), None)
    grid_lon_line = next((line for line in lines if 'Grid_longitude:' in line), None)
    grid_height_line = next((line for line in lines if 'Grid_height:' in line), None)
    indicators_line = next((line for line in lines if 'Indicator:' in line), None)

    # Parse these lines to get station metadata
    station_metadata = {}
    if all([grid_idx_line, grid_lat_line, grid_lon_line, grid_height_line, indicators_line]):
        indicators = indicators_line.split(';')[1:-1]  # Skip 'Indicator:' and empty last element
        latitudes = grid_lat_line.split(';')[1:-1]  # Skip 'Grid_latitude:' and empty last element
        longitudes = grid_lon_line.split(';')[1:-1]  # Skip 'Grid_longitude:' and empty last element
        heights = grid_height_line.split(';')[1:-1]  # Skip 'Grid_height:' and empty last element

        # Create station metadata
        station_metadata = {
            'elevation': {stn: float(h) for stn, h in zip(indicators, heights) if stn.strip()},
            'latitude': {stn: float(lat) for stn, lat in zip(indicators, latitudes) if stn.strip()},
            'longitude': {stn: float(lon) for stn, lon in zip(indicators, longitudes) if stn.strip()},
            'model_height_difference': {stn: 0.0 for stn in indicators if stn.strip()},  # Placeholder
            'id': {stn: i for i, stn in enumerate(indicators) if stn.strip()}
        }
    else:
        LOGGER.warning("Could not extract station metadata from file")
        # Create a basic station list
        unique_stations = df['stn'].unique().to_list()
        station_metadata = {
            'elevation': {stn: 0.0 for stn in unique_stations},
            'latitude': {stn: 0.0 for stn in unique_stations},
            'longitude': {stn: 0.0 for stn in unique_stations},
            'model_height_difference': {stn: 0.0 for stn in unique_stations},
            'id': {stn: i for i, stn in enumerate(unique_stations)}
        }

    # Estimate surface pressure using barometric formula and station elevation
    # This is a simplification - the paper may use actual pressure data
    LOGGER.info("Calculating surface pressure")

    def calculate_pressure(elevation):
        """Calculate pressure from elevation using barometric formula"""
        p0 = 1013.25  # sea level standard atmospheric pressure in hPa
        g = 9.80665  # gravitational acceleration (m/s^2)
        M = 0.0289644  # molar mass of Earth's air (kg/mol)
        R = 8.31447  # universal gas constant (J/(mol·K))
        T = 288.15  # standard temperature (K)

        # Barometric formula
        return p0 * np.exp(-(g * M * elevation) / (R * T))

    # Get elevations for each station in the data
    station_elevations = {stn: station_metadata['elevation'].get(stn, 0.0) for stn in
                          features_df['stn'].unique().to_list()}

    # Add pressure to features_df
    features_df = features_df.with_columns([
        pl.col('stn').map_dict(station_elevations).alias('elevation')
    ])

    features_df = features_df.with_columns([
        pl.Series('coe:surface_air_pressure_ensavg',
                  calculate_pressure(features_df['elevation'].to_numpy()))
    ])

    # Calculate water vapor mixing ratio
    LOGGER.info("Calculating water vapor mixing ratio")

    # Extract the needed data
    t_data = features_df.select('coe:air_temperature_ensavg').to_numpy().flatten()
    td_data = features_df.select('coe:dew_point_temperature_ensavg').to_numpy().flatten()
    p_data = features_df.select('coe:surface_air_pressure_ensavg').to_numpy().flatten()

    # Calculate mixing ratio
    def calculate_mixing_ratio(t, td, p):
        is_positive = t >= 0
        a = np.where(is_positive, 17.368, 17.856)
        b = np.where(is_positive, 238.83, 245.52)
        c = np.where(is_positive, 6.107, 6.108)

        e = c * np.exp((a * td) / (b + td))
        return 622.0 * (e / (p - e))

    mixing_ratio = calculate_mixing_ratio(t_data, td_data, p_data)

    # Add mixing ratio to features_df
    features_df = features_df.with_columns([
        pl.Series('coe:water_vapor_mixing_ratio_ensavg', mixing_ratio)
    ])

    # Add time information features
    LOGGER.info("Adding time features")

    # Extract time components
    time_day_of_year = features_df.select(pl.col('time').dt.ordinal_day()).to_numpy().flatten()
    time_hour_of_day = features_df.select(pl.col('time').dt.hour()).to_numpy().flatten()

    # Add time features
    features_df = features_df.with_columns([
        pl.Series('time:cos_dayofyear', np.cos(2 * np.pi * time_day_of_year / 366)),
        pl.Series('time:sin_dayofyear', np.sin(2 * np.pi * time_day_of_year / 366)),
        pl.Series('time:cos_hourofday', np.cos(2 * np.pi * time_hour_of_day / 24)),
        pl.Series('time:sin_hourofday', np.sin(2 * np.pi * time_hour_of_day / 24)),
        pl.col('leadtime_hours').alias('coe:leadtime')
    ])

    # Convert to xarray Dataset
    LOGGER.info("Converting to xarray")

    # Drop unnecessary columns
    features_df = features_df.drop(['elevation'])

    # Create forecast_reference_time and t dimensions
    # forecast_reference_time is the time of the forecast
    # t is the leadtime in hours
    features_xr = (
        features_df
        .with_columns([
            pl.col('time').alias('forecast_reference_time'),
            pl.duration(hours=pl.col('leadtime_hours')).alias('t')
        ])
        .drop(['time', 'leadtime_hours'])
        .to_pandas()
        .set_index(['forecast_reference_time', 't', 'stn'])
        .to_xarray()
    )

    # For this example, we'll simulate targets
    # In a real scenario, you would use actual observations
    LOGGER.info("Creating simulated targets (replace with real observations)")

    # Create simulated targets by adding noise to forecasts
    np.random.seed(42)  # For reproducibility

    # Extract base variables
    temp = features_xr['coe:air_temperature_ensavg'].values
    dewpoint = features_xr['coe:dew_point_temperature_ensavg'].values
    pressure = features_xr['coe:surface_air_pressure_ensavg'].values
    relhum = features_xr['coe:relative_humidity_ensavg'].values
    mixratio = features_xr['coe:water_vapor_mixing_ratio_ensavg'].values

    # Create targets dataset with simulated observations
    targets_data = {
        'obs:air_temperature': (
        ['forecast_reference_time', 't', 'stn'], temp + np.random.normal(0, 0.5, size=temp.shape)),
        'obs:dew_point_temperature': (
        ['forecast_reference_time', 't', 'stn'], dewpoint + np.random.normal(0, 0.5, size=dewpoint.shape)),
        'obs:surface_air_pressure': (
        ['forecast_reference_time', 't', 'stn'], pressure + np.random.normal(0, 1.0, size=pressure.shape)),
        'obs:relative_humidity': (
        ['forecast_reference_time', 't', 'stn'], np.clip(relhum + np.random.normal(0, 2.0, size=relhum.shape), 0, 100)),
        'obs:water_vapor_mixing_ratio': (
        ['forecast_reference_time', 't', 'stn'], mixratio + np.random.normal(0, 0.2, size=mixratio.shape))
    }

    targets_xr = xr.Dataset(
        targets_data,
        coords=features_xr.coords
    )

    # Add station metadata to targets
    for key, value_dict in station_metadata.items():
        if key != 'id':  # We'll handle ID separately
            targets_xr[key] = ('stn', [value_dict.get(s, 0.0) for s in targets_xr.stn.values])

    # Add owner_id (required by original code)
    targets_xr['owner_id'] = ('stn', np.ones(len(targets_xr.stn), dtype=int))

    # Filter stations with completeness threshold
    targets_xr = filter_stations(targets_xr)

    # Update features to match filtered targets
    features_xr = features_xr.reindex_like(targets_xr)

    # Reshape and drop missing values as in original code
    x, y = reshape(features_xr, targets_xr)
    x, y = drop_missing(x, y)

    # Get station ID mapping as in original code
    station_id_map = {s: i for i, s in enumerate(targets_xr.station.values)}
    station_metadata['id'] = station_id_map
    station_id_coord = x.station.to_pandas().map(station_id_map).values

    # Reset coords and add station_id
    x = x.reset_coords(["owner_id", "elevation", "longitude", "latitude", "model_height_difference"], drop=True)
    y = y.reset_coords(["owner_id", "elevation", "longitude", "latitude", "model_height_difference"], drop=True)
    x = x.assign_coords(station_id=("s", station_id_coord))

    # Save the data
    x_path = output_dir / "features.zarr"
    y_path = output_dir / "targets.zarr"
    stations_path = output_dir / "stations_list.json"

    LOGGER.info(f"Saving features to {x_path}")
    x.to_dataset("var").to_zarr(x_path, mode='w')

    LOGGER.info(f"Saving targets to {y_path}")
    y.to_dataset("var").to_zarr(y_path, mode='w')

    LOGGER.info(f"Saving station metadata to {stations_path}")
    with open(stations_path, 'w') as f:
        json.dump(station_metadata, f, indent=4)

    LOGGER.info("Preprocessing complete")

    return {
        'features': str(x_path),
        'targets': str(y_path),
        'stations_list': str(stations_path)
    }


# Helper functions copied from original code
def reshape(features: xr.Dataset, targets: xr.Dataset) -> tuple[xr.DataArray]:
    """Reshape data to 2-d (sample, variable) tensors."""
    dims = ["forecast_reference_time", "t", "station"]
    x = (
        features.to_array("var")
        .stack(s=dims, create_index=False)
        .transpose("s", ..., "var")
    )
    y = (
        targets.to_array("var")
        .stack(s=dims, create_index=False)
        .transpose("s", ..., "var")
    )
    LOGGER.info(f"Reshaped: x -> {dict(x.sizes)} and y -> {dict(y.sizes)}")
    return x, y


def drop_missing(x: xr.DataArray, y: xr.DataArray) -> tuple[xr.DataArray]:
    """Only keep complete (all features and targets available) samples."""
    n_samples = len(x.s.values)
    mask_x_dims = [dim for dim in x.dims if dim != "s"]
    mask_y_dims = [dim for dim in y.dims if dim != "s"]
    x = x[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(x).all(dim=mask_x_dims)]
    x = x[np.isfinite(x).all(dim=mask_x_dims)]
    n_incomplete_samples = n_samples - len(x.s.values)
    LOGGER.info(f"Dropped {n_incomplete_samples} incomplete samples out of {n_samples}")
    return x, y


# Usage example
if __name__ == "__main__":
    # Update with your file path
    filepath = "D:\ML2025\PCPP\data\COSMO-E-all-stations.csv"
    output_dir = "data/01_raw"

    inputs = preprocess_cosmo_data(filepath, output_dir)
    print(f"Generated input files: {inputs}")