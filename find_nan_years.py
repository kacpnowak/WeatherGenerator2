import zarr
import numpy as np

nan_years = []
for year in range(2000, 2210):
    try:
        g = zarr.open_group(f'/e/data1/climateai/hclimrep/data/ifs-fesom/ocean_node/ocean_node_{year}.zarr', mode='r')
        data = g['data']
        nodes = data.attrs.get('n_points', data.attrs.get('nod2', 0))
        chunk = data[0:nodes, :]
        if np.isnan(chunk).all():
            nan_years.append(year)
    except Exception as e:
        print(f"Year {year} error: {e}")

print(f"NaN years: {nan_years}")
