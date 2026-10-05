import os

from oceanbench.core import input_datasets as obi


def get_nowcast(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "glo12_nowcast_full.nc")

    print("Loading the complete GLO12 nowcast benchmark dataset from oceanbench...")
    try:
        # This built-in function automatically fetches ALL dates
        # hosted on the oceanbench server (weekly data for 2024).
        glo12 = obi.glo12_nowcasts()

        print(f"Successfully loaded dataset with {len(glo12.time)} timesteps!")
        print(f"Saving to {out_file}...")

        # Save the entire combined dataset to a single file
        glo12.to_netcdf(out_file)
        print("Done!")

    except Exception as e:
        print(f"Failed to get data: {type(e).__name__} - {e}")


if __name__ == "__main__":
    out_dir = "/e/data1/climateai/hclimrep/data/glorys_forcings"
    get_nowcast(out_dir)
