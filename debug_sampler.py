import zarr
import dask.array as da
import glob
import numpy as np
import datetime

filenames = sorted(glob.glob('/e/data1/climateai/hclimrep/data/ifs-fesom/ocean_node/*'))
print(f"Found {len(filenames)} files.")

# Load dates
s_groups = [zarr.open_group(name, mode="r") for name in filenames]
s_times = [da.from_zarr(group["dates"]) for group in s_groups]
source_time = da.concatenate(s_times, axis=0)
start_source = source_time[0][0].compute()
end_source = source_time[-1][0].compute()

print(f"Start: {np.datetime64(start_source, 's')}")
print(f"End: {np.datetime64(end_source, 's')}")

# Simulate idx=23907
# Training start_date is 2000-01-01
t_start = np.datetime64('2000-01-01T00:00:00')
time_step = np.timedelta64(86400, 's') # 24h
idx = 23907

window_start = t_start + idx * time_step
# In data_sampler, if it needs forecast steps, it might ask for window_end = window_start + (num_steps * time_step)
num_forecast_steps = 3
window_end = window_start + num_forecast_steps * time_step

print(f"Window Start: {window_start}")
print(f"Window End: {window_end}")

# Check if within bounds
if (window_end < start_source or window_start > end_source):
    print("Out of overall bounds!")

# delta_t
delta_t_start = window_start - start_source
source_period = (source_time[s_groups[0]["data"].attrs.get("n_points", s_groups[0]["data"].attrs.get("nod2"))][0] - source_time[0][0]).compute()
print(f"Source period: {source_period}")
start_didx = delta_t_start // source_period
end_didx = (window_end - start_source) // source_period

print(f"Start didx: {start_didx}, End didx: {end_didx}")
