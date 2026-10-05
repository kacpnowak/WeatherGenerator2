import fsspec
import xarray as xr
mapper = fsspec.get_mapper("reference://", fo="/e/data1/climateai/hclimrep/data/glorys/glorys_full_mesh.parq", remote_protocol="file")
ds = xr.open_dataset(mapper, engine="zarr", consolidated=False)
print("Start:", ds.time.values[0])
print("End:", ds.time.values[-1])


