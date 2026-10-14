import glob
import json
import logging
from pathlib import Path
from typing import override

import dask
import dask.array as da
import numpy as np
import zarr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DTRange,
    NDArray,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    t_epsilon,
)
from weathergen.readers_extra.fesom_mesh import bathymetry, dry_mask, read_mesh_entity
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


def apply_normalization_overrides(
    colnames: list[str],
    mean: NDArray,
    std: NDArray,
    overrides: dict,
) -> tuple[NDArray, NDArray]:
    """
    Replace the normalisation statistics of selected channels.

    Parameters
    ----------
    colnames :
        channel names, in the order in which `mean` and `std` are indexed
    mean, std :
        statistics per column of `colnames`; they are not modified
    overrides :
        {channel: {"mean": float, "std": float}}. Keys starting with "_" (metadata) and channels
        that are not in `colnames` (a stream may select fewer channels than the file covers)
        are ignored.

    Returns
    -------
    Copies of `mean` and `std` with the overridden channels replaced.

    Raises
    ------
    ValueError
        if `colnames`, `mean` and `std` differ in length, or an applied override has a
        non-positive or non-finite std or a non-finite mean.
    """
    new_mean = np.array(mean, dtype=np.float64)
    new_std = np.array(std, dtype=np.float64)
    if not len(colnames) == len(new_mean) == len(new_std):
        raise ValueError(
            f"{len(colnames)} column names but {len(new_mean)} means and {len(new_std)} stds"
        )

    column = {name: i for i, name in enumerate(colnames)}
    for name, stats in overrides.items():
        if name.startswith("_") or name not in column:
            continue
        ch_mean, ch_std = float(stats["mean"]), float(stats["std"])
        if not (np.isfinite(ch_std) and ch_std > 0.0):
            raise ValueError(f"Normalisation override for {name!r} needs std > 0, got {ch_std}")
        if not np.isfinite(ch_mean):
            raise ValueError(f"Normalisation override for {name!r} has a non-finite mean")
        new_mean[column[name]] = ch_mean
        new_std[column[name]] = ch_std

    return new_mean, new_std


class DataReaderFesom(DataReaderTimestep):
    """
    A dataset class for handling temporal windows of FESOM model output data stored in Zarr format.

    This class is optimized for use with multiple dataloader workers by implementing
    lazy initialization of file handles and efficient, batched data reads.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        # Store configuration but DO NOT open files here
        self.filenames = sorted(glob.glob(str(filename) + "/*"))
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self.target_files = self.filenames

        self._src_lat_conv = False
        self._src_lon_conv = False
        self._trg_lat_conv = False
        self._trg_lon_conv = False

        if "target_file" in stream_info:
            self.target_files = sorted(glob.glob(str(stream_info["target_file"]) + "/*"))

        if len(self.filenames) == 0:
            self.init_empty()
            self._initialized = True
            return

        # Initialize data-dependent attributes to None. They will be set by _lazy_init.
        self.source_time: da.Array | None = None
        self.source_data: da.Array | None = None
        self.target_time: da.Array | None = None
        self.target_data: da.Array | None = None
        self.len = 0  # Default length is 0 until initialized
        self.source_channels = []
        self.source_idx = []
        self.target_channels = []
        self.target_idx = []
        # the sampler sizes tokens from get_geoinfo_size() before any data is read, so the
        # geoinfo channels are fixed here from the stream config
        self._wet_mask_from_mesh = bool(stream_info.get("wet_mask_from_mesh"))
        self._bathymetry_geoinfo = bool(stream_info.get("bathymetry_geoinfo"))
        if (self._wet_mask_from_mesh or self._bathymetry_geoinfo) and not stream_info.get(
            "mesh_path"
        ):
            raise ValueError(
                f"Stream {stream_info.get('name')!r}: wet_mask_from_mesh and bathymetry_geoinfo "
                "need the stream option mesh_path"
            )
        self.geoinfo_channels = ["bathymetry"] if self._bathymetry_geoinfo else []
        self.geoinfo_idx = [0] if self._bathymetry_geoinfo else []
        self.properties = {}
        self.fake_specs = {}
        self.fake_target = False

        if len(self.filenames) == 0 or len(self.target_files) == 0:
            name = stream_info["name"]
            _logger.warning(
                f"{name} couldn't find any files matching {filename}. Stream is skipped."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            # No need to return, the length is 0, so it will be skipped.

        # We call super() last, after we know if the stream is valid or not.
        # We also pass dummy values, as the real ones will be set in _lazy_init.
        super().__init__(self._tw_handler, self._stream_info)

        # This flag ensures initialization happens only once per worker
        self._initialized = False
        if len(self.filenames) > 0 and len(self.target_files) > 0:
            # We need to initialize the channels in __init__ so that the
            # MultiStreamDataSampler can correctly identify if a stream is forcing or not.
            s_group = zarr.open_group(self.filenames[0], mode="r")
            t_group = zarr.open_group(self.target_files[0], mode="r")

            self.source_mesh_size = self._get_mesh_size(s_group)
            self.target_mesh_size = self._get_mesh_size(t_group)

            source_colnames: list[str] = list(s_group["data"].attrs["colnames"])
            target_colnames: list[str] = list(t_group["data"].attrs["colnames"])

            source_cols_idx = list(np.arange(len(source_colnames), dtype=int))
            target_cols_idx = list(np.arange(len(target_colnames), dtype=int))

            src_lat_index: int = source_colnames.index("lat")
            src_lon_index: int = source_colnames.index("lon")
            trg_lat_index: int = target_colnames.index("lat")
            trg_lon_index: int = target_colnames.index("lon")

            source_colnames = self._remove_lonlat(source_colnames)
            target_colnames = self._remove_lonlat(target_colnames)

            source_cols_idx.remove(src_lat_index)
            source_cols_idx.remove(src_lon_index)
            source_cols_idx = np.array(source_cols_idx)

            target_cols_idx.remove(trg_lat_index)
            target_cols_idx.remove(trg_lon_index)
            target_cols_idx = np.array(target_cols_idx)

            source_channels = self._stream_info.get("source")
            source_excl = self._stream_info.get("source_exclude")
            self.source_channels, self.source_idx = (
                self.select(source_colnames, source_cols_idx, source_channels, source_excl)
                if source_channels or source_excl
                else (source_colnames, source_cols_idx)
            )

            target_channels = self._stream_info.get("target")
            target_excl = self._stream_info.get("target_exclude")
            self.target_channels, self.target_idx = (
                self.select(target_colnames, target_cols_idx, target_channels, target_excl)
                if target_channels or target_excl
                else (target_colnames, target_cols_idx)
            )

            self.target_channel_weights = self.parse_target_channel_weights()

    def _get_mesh_size(self, group: zarr.Group) -> int:
        if "n_points" in group["data"].attrs:
            return group["data"].attrs["n_points"]
        else:
            return group["data"].attrs["nod2"]

    def _read_slice(self, arr_list, cum_sizes, start_row, end_row):
        import dask.array as da
        start_f_idx = np.searchsorted(cum_sizes, start_row, side='right') - 1
        end_f_idx = np.searchsorted(cum_sizes, end_row - 1, side='right') - 1
        
        if start_f_idx == end_f_idx:
            l_start = start_row - cum_sizes[start_f_idx]
            l_end = end_row - cum_sizes[start_f_idx]
            return arr_list[start_f_idx][l_start:l_end]
        else:
            chunks = []
            l_start = start_row - cum_sizes[start_f_idx]
            chunks.append(arr_list[start_f_idx][l_start:])
            for i in range(start_f_idx + 1, end_f_idx):
                chunks.append(arr_list[i])
            l_end = end_row - cum_sizes[end_f_idx]
            chunks.append(arr_list[end_f_idx][:l_end])
            return da.concatenate(chunks, axis=0)

    def _reorder_groups(self, colnames: list[str], groups: list[zarr.Group]) -> list[da.Array]:
        reordered_data_arrays: list[da.Array] = []

        for group in groups:
            local_colnames = group["data"].attrs["colnames"]

            # If the order is already correct, no need to do anything.
            if local_colnames == colnames:
                reordered_data_arrays.append(da.from_zarr(group["data"]))
            else:
                # Create the list of indices to re-shuffle the columns.
                reorder_indices = [local_colnames.index(name) for name in colnames]

                # Lazily re-index the dask array. This operation is not executed immediately.
                dask_array = da.from_zarr(group["data"])
                reordered_array = dask_array[:, reorder_indices]
                reordered_data_arrays.append(reordered_array)

        return reordered_data_arrays

    def _remove_lonlat(self, colnames: list[str]) -> list[str]:
        temp_colnames = list(colnames)
        temp_colnames.remove("lat")
        temp_colnames.remove("lon")
        return temp_colnames

    def _lazy_init(self) -> None:
        """
        Initializes the dataset object. This method is called once per worker process
        to ensure dask scheduler is not shared between them.
        """
        # pylint: disable=attribute-defined-outside-init

        if self._initialized:
            return

        _logger.info(f"Initialising {self._stream_info['name']}")

        # Each worker now opens its own file handles safely
        s_groups: list[zarr.Group] = [zarr.open_group(name, mode="r") for name in self.filenames]
        t_groups: list[zarr.Group] = [zarr.open_group(name, mode="r") for name in self.target_files]

        # Store lists of dask arrays instead of concatenating them to avoid massive Dask graphs
        self.s_times_list = [da.from_zarr(group["dates"]) for group in s_groups]
        self.t_times_list = [da.from_zarr(group["dates"]) for group in t_groups]
        
        self.s_times_cum = np.cumsum([0] + [arr.shape[0] for arr in self.s_times_list])
        self.t_times_cum = np.cumsum([0] + [arr.shape[0] for arr in self.t_times_list])

        # Use the first group for metadata
        self.source_mesh_size = self._get_mesh_size(s_groups[0])
        self.target_mesh_size = self._get_mesh_size(t_groups[0])

        # Metadata reading is cheap, but let's do it with the rest of the init
        self.start_source = self.s_times_list[0][0][0].compute()
        self.end_source = self.s_times_list[-1][-1][0].compute()
        
        self.start_target = self.t_times_list[0][0][0].compute()
        self.end_target = self.t_times_list[-1][-1][0].compute()

        if self.start_source > self._tw_handler.t_end or self.end_source < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(
                f"Stream {name} source lies completely outside the time window, "
                f"no stream data will be generated. {self.start_source} - {self.end_source} "
                f"vs {self._tw_handler.t_start} - {self._tw_handler.t_end}"
            )
            self.init_empty()
            self._initialized = True
            return



        if self.start_target > self._tw_handler.t_end or self.end_target < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            self._initialized = True
            return

        self.source_period = (
            self.s_times_list[0][self.source_mesh_size][0] - self.s_times_list[0][0][0]
        ).compute()
        self.target_period = (
            self.t_times_list[0][self.target_mesh_size][0] - self.t_times_list[0][0][0]
        ).compute()

        # Re-initialize the parent class with correct time info
        super().__init__(  # Initialise only for source as source-target split is not supported
            self._tw_handler,
            self._stream_info,
            self.start_source,
            self.end_source,
            self.source_period,
        )

        if (
            self._tw_handler.t_start > self.start_source
            and self._tw_handler.t_start > self.end_source
        ):
            self.source_start_idx = (
                (self._tw_handler.t_start - self.start_source) // self.source_period + 1
            ) * self.source_mesh_size
        else:
            self.source_start_idx = 0

        if (
            self._tw_handler.t_start > self.start_target
            and self._tw_handler.t_start > self.end_target
        ):
            self.target_start_idx = (
                (self._tw_handler.t_start - self.start_target) // self.target_period + 1
            ) * self.target_mesh_size
        else:
            self.target_start_idx = 0

        self.source_end_idx = (
            (self._tw_handler.t_end - self.start_source) // self.source_period + 1
        ) * self.source_mesh_size
        self.target_end_idx = (
            (self._tw_handler.t_end - self.start_target) // self.target_period + 1
        ) * self.target_mesh_size

        if self.source_end_idx > self.s_times_cum[-1]:
            self.source_end_idx = int(self.s_times_cum[-1])
        if self.target_end_idx > self.t_times_cum[-1]:
            self.target_end_idx = int(self.t_times_cum[-1])

        self.source_len = (self.source_end_idx - self.source_start_idx) // self.source_mesh_size
        self.target_len = (self.target_end_idx - self.target_start_idx) // self.target_mesh_size
        self.len = min(self.source_len, self.target_len)

        # Check for a valid length after calculations
        if self.len <= 0:
            self.init_empty()
            self._initialized = True
            return

        self.source_colnames: list[str] = list(s_groups[0]["data"].attrs["colnames"])
        self.target_colnames: list[str] = list(t_groups[0]["data"].attrs["colnames"])
        # means / std below are indexed like these full lists (lon and lat included)
        src_colnames_full = list(self.source_colnames)
        trg_colnames_full = list(self.target_colnames)

        self.source_cols_idx = list(np.arange(len(self.source_colnames), dtype=int))
        self.target_cols_idx = list(np.arange(len(self.target_colnames), dtype=int))

        self.src_lat_index: int = self.source_colnames.index("lat")
        self.src_lon_index: int = self.source_colnames.index("lon")
        self.trg_lat_index: int = self.target_colnames.index("lat")
        self.trg_lon_index: int = self.target_colnames.index("lon")

        self.s_data_list = self._reorder_groups(self.source_colnames, s_groups)
        self.t_data_list = self._reorder_groups(self.target_colnames, t_groups)
        
        self.s_data_cum = np.cumsum([0] + [arr.shape[0] for arr in self.s_data_list])
        self.t_data_cum = np.cumsum([0] + [arr.shape[0] for arr in self.t_data_list])

        # Modify a copy, not the original list while iterating
        self.source_colnames = self._remove_lonlat(self.source_colnames)
        self.target_colnames = self._remove_lonlat(self.target_colnames)

        self.source_cols_idx.remove(self.src_lat_index)
        self.source_cols_idx.remove(self.src_lon_index)
        self.source_cols_idx = np.array(self.source_cols_idx)

        self.target_cols_idx.remove(self.trg_lat_index)
        self.target_cols_idx.remove(self.trg_lon_index)
        self.target_cols_idx = np.array(self.target_cols_idx)

        self.properties = {"stream_id": s_groups[0]["data"].attrs["obs_id"]}

        self.source_mean = np.concatenate(
            (np.array([0, 0]), np.array(s_groups[0]["data"].attrs["means"]))
        )
        self.source_stdev = np.concatenate(
            (np.array([1, 1]), np.array(s_groups[0]["data"].attrs["std"]))
        )
        self.source_stdev[self.source_stdev <= 1e-5] = 1.0

        self.target_mean = np.concatenate(
            (np.array([0, 0]), np.array(t_groups[0]["data"].attrs["means"]))
        )
        self.target_stdev = np.concatenate(
            (np.array([1, 1]), np.array(t_groups[0]["data"].attrs["std"]))
        )
        self.target_stdev[self.target_stdev <= 1e-5] = 1.0

        self._override_normalization(src_colnames_full, trg_colnames_full)


        source_channels = self._stream_info.get("source")
        source_excl = self._stream_info.get("source_exclude")
        self.source_channels, self.source_idx = (
            self.select(self.source_colnames, self.source_cols_idx, source_channels, source_excl)
            if source_channels or source_excl
            else (self.source_colnames, self.source_cols_idx)
        )

        target_channels = self._stream_info.get("target")
        target_excl = self._stream_info.get("target_exclude")
        self.target_channels, self.target_idx = (
            self.select(self.target_colnames, self.target_cols_idx, target_channels, target_excl)
            if target_channels or target_excl
            else (self.target_colnames, self.target_cols_idx)
        )

        src_timestep_lats = self.s_data_list[0][: self.source_mesh_size, self.src_lat_index].compute()
        trg_timestep_lats = self.t_data_list[0][: self.target_mesh_size, self.trg_lat_index].compute()

        if np.any(src_timestep_lats > 90.0):
            _logger.warning(
                f"Latitude for stream '{self._stream_info['name']}' "
                f"source appears to be in a [0, 180] format. "
                f"It will be automatically converted to the required [-90, 90] format."
            )
            self._src_lat_conv = True

        if np.any(trg_timestep_lats > 90.0):
            _logger.warning(
                f"Latitude for stream '{self._stream_info['name']}' "
                f"target appears to be in a [0, 180] format. "
                f"It will be automatically converted to the required [-90, 90] format."
            )
            self._trg_lat_conv = True

        src_timestep_lons = self.s_data_list[0][: self.source_mesh_size, self.src_lon_index].compute()
        trg_timestep_lons = self.t_data_list[0][: self.target_mesh_size, self.trg_lon_index].compute()

        if np.any(src_timestep_lons > 180.0):
            _logger.warning(
                f"Longitude for stream '{self._stream_info['name']}' "
                f"source appears to be in a [0, 360] format. "
                f"It will be automatically converted to the required [-180, 180] format."
            )
            self._src_lon_conv = True

        if np.any(trg_timestep_lons > 180.0):
            _logger.warning(
                f"Longitude for stream '{self._stream_info['name']}' "
                f"target appears to be in a [0, 360] format."
                f"It will be automatically converted to the required [-180, 180] format."
            )
            self._trg_lon_conv = True

        if self._src_lat_conv: src_timestep_lats = 90.0 - src_timestep_lats
        if self._trg_lat_conv: trg_timestep_lats = 90.0 - trg_timestep_lats
        if self._src_lon_conv: src_timestep_lons = ((src_timestep_lons + 180.0) % 360.0) - 180.0
        if self._trg_lon_conv: trg_timestep_lons = ((trg_timestep_lons + 180.0) % 360.0) - 180.0

        self.cached_source_coords = np.stack([src_timestep_lats, src_timestep_lons], axis=1)
        self.cached_target_coords = np.stack([trg_timestep_lats, trg_timestep_lons], axis=1)

        if not self._bathymetry_geoinfo:
            self.geoinfo_channels = []
            self.geoinfo_idx = []

        self.src_mean_vec = self.source_mean[self.source_idx]
        self.src_stdev_vec = self.source_stdev[self.source_idx]
        self.trg_mean_vec = self.target_mean[self.target_idx]
        self.trg_stdev_vec = self.target_stdev[self.target_idx]

        if self._wet_mask_from_mesh or self._bathymetry_geoinfo:
            assert self.target_mesh_size == self.source_mesh_size, (
                "mesh-based masking needs source and target on the same mesh"
            )
            self._mesh = read_mesh_entity(self._stream_info["mesh_path"], self.source_mesh_size)
        if self._wet_mask_from_mesh:
            # one-day masks in mesh order; tiled per time step in get_source / get_target
            self._dry_src_day = dry_mask(self.source_channels, self._mesh, 1)
            self._dry_trg_day = dry_mask(self.target_channels, self._mesh, 1)
        if self._bathymetry_geoinfo:
            raw = bathymetry(self._mesh)
            self._geo_day = raw[:, None].astype(np.float32)
            self.mean_geoinfo = np.array([raw.mean()])
            self.stdev_geoinfo = np.array([raw.std()])

        self._initialized = True

    def _override_normalization(
        self, src_colnames_full: list[str], trg_colnames_full: list[str]
    ) -> None:
        """
        Replace the archive statistics by those of the `normalization_stats` stream option.

        The option is the path of a JSON file {channel: {"mean": ..., "std": ...}} (see
        compute_wet_stats.py). The archive statistics include the zeros FESOM stores below the
        sea floor; the file holds statistics over wet points only. `src_colnames_full` and
        `trg_colnames_full` are the archive column names (lon and lat included), which is how
        source_mean / target_mean are indexed.
        """
        stats_path = self._stream_info.get("normalization_stats")
        if not stats_path:
            return

        with open(stats_path) as f:
            overrides = json.load(f)

        self.source_mean, self.source_stdev = apply_normalization_overrides(
            src_colnames_full, self.source_mean, self.source_stdev, overrides
        )
        self.target_mean, self.target_stdev = apply_normalization_overrides(
            trg_colnames_full, self.target_mean, self.target_stdev, overrides
        )

        src_changed = [c for c in src_colnames_full if c in overrides and not c.startswith("_")]
        trg_changed = [c for c in trg_colnames_full if c in overrides and not c.startswith("_")]
        _logger.info(
            f"{self._stream_info['name']}: normalisation statistics of {len(src_changed)} source "
            f"and {len(trg_changed)} target channels replaced from {stats_path}: "
            f"{', '.join(trg_changed)}"
        )

    def select(
        self,
        colnames: list[str],
        cols_idx: NDArray,
        ch_filters: list[str] | None,
        excl: list[str] | None = None,
    ) -> tuple[list[str], NDArray]:
        if excl and ch_filters:
            mask = [
                any(f == c for f in ch_filters) and all(ex not in c for ex in excl)
                for c in colnames
            ]
        elif ch_filters:
            mask = [any(f == c for f in ch_filters) for c in colnames]
        elif excl:
            mask = [all(ex not in c for ex in excl) for c in colnames]
        else:
            assert False, "Cannot use select with both ch_filters and excl as None"

        selected_cols_idx = cols_idx[np.where(mask)[0]]
        selected_colnames = [colnames[i] for i in np.where(mask)[0]]
        return selected_colnames, selected_cols_idx

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        # the model was sized with the bathymetry geoinfo channel; an empty reader must keep
        # the same token width
        if self._stream_info.get("bathymetry_geoinfo"):
            self.geoinfo_channels = ["bathymetry"]
            self.geoinfo_idx = [0]

    @override
    def length(self) -> int:
        # Make sure initialization has happened before returning length
        self._lazy_init()
        return self.len

    def _get_source_idxs(self, idx: TIndex) -> tuple[NDArray, DTRange]:
        """
        Get source dataset indexes for a given time window index, when the dataset is periodic.

        This function assumes state of a variable is persistent, thus if no data is found
        in the time window, last measurement is used before the beggining of the windows is used.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int64]
            Array of dataset indexes corresponding to the time window.
        """
        tw_handler = self.time_window_handler

        # Function is separated from the class to allow testing without instantiating the class.
        dtr = tw_handler.window(idx)
        # If there is no or only marginal overlap with the dataset, return empty index ranges
        if (
            not self.start_source
            or not self.end_source
            or dtr.end < self.start_source
            or dtr.start > self.end_source
            or dtr.start < self.start_source
            or dtr.end > self.end_source
            or (self.end_source is not None and dtr.start > self.end_source)
        ):
            return (np.array([], dtype=np.int64), dtr)

        # relative time in dataset
        delta_t_start = dtr.start - self.start_source
        delta_t_end = dtr.end - self.start_source - t_epsilon
        assert isinstance(delta_t_start, np.timedelta64), "delta_t_start must be timedelta64"
        start_didx = delta_t_start // self.source_period
        end_didx = delta_t_end // self.source_period

        # adjust start_idx if not exactly on start time
        if (delta_t_start % self.source_period) > np.timedelta64(0, "s"):
            # empty window in between two timesteps
            if start_didx == end_didx:
                return (np.array([start_didx], dtype=np.int64), dtr)
            start_didx += 1

        end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / self.source_period)
        return (np.arange(start_didx, end_didx + 1, dtype=np.int64), dtr)

    def _get_target_idxs(self, idx: TIndex) -> tuple[NDArray, DTRange]:
        """
        Get target dataset indexes for a given time window index, when the dataset is periodic.

        This function assumes state of a variable is persistent, thus if no data is found
        in the time window, last measurement is used before the beggining of the windows is used.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int64]
            Array of dataset indexes corresponding to the time window.
        """
        tw_handler = self.time_window_handler

        # Function is separated from the class to allow testing without instantiating the class.
        dtr = tw_handler.window(idx)
        # If there is no or only marginal overlap with the dataset, return empty index ranges
        if (
            not self.start_target
            or not self.end_target
            or dtr.end < self.start_target
            or dtr.start > self.end_target
            or dtr.start < self.start_target
            or dtr.end > self.end_target
            or (self.end_target is not None and dtr.start > self.end_target)
        ):
            return (np.array([], dtype=np.int64), dtr)

        # relative time in dataset
        delta_t_start = dtr.start - self.start_target
        delta_t_end = dtr.end - self.start_target - t_epsilon
        assert isinstance(delta_t_start, np.timedelta64), "delta_t_start must be timedelta64"
        start_didx = delta_t_start // self.target_period
        end_didx = delta_t_end // self.target_period

        # adjust start_idx if not exactly on start time
        if (delta_t_start % self.target_period) > np.timedelta64(0, "s"):
            # empty window in between two timesteps
            if start_didx == end_didx:
                return (np.array([start_didx], dtype=np.int64), dtr)
            start_didx += 1

        end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / self.target_period)
        return (np.arange(start_didx, end_didx + 1, dtype=np.int64), dtr)

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        self._lazy_init()
        (t_idxs, dtr) = self._get_source_idxs(idx)
        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(self.source_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = t_idxs[0] * self.source_mesh_size
        end_row = (t_idxs[-1] + 1) * self.source_mesh_size

        # Note: we read all columns from start_row to end_row once,
        # then select the ones we need. This is more efficient for Zarr.
        full_data_slice = self._read_slice(self.s_data_list, self.s_data_cum, start_row, end_row)
        datetimes_lazy = self._read_slice(self.s_times_list, self.s_times_cum, start_row, end_row)

        # Define the specific slices we need from the larger block
        data_lazy = full_data_slice[:, self.source_idx]

        # Dask optimizes this to a single (or few) efficient read operation(s).
        data, datetimes = dask.compute(
            data_lazy, datetimes_lazy, scheduler="single-threaded"
        )
        
        n_steps = data.shape[0] // self.source_mesh_size
        coords = np.tile(self.cached_source_coords, (n_steps, 1))
        if self._wet_mask_from_mesh:
            # rows are still in day-major mesh order here. Dry inputs become the channel mean,
            # i.e. 0 after normalisation
            dry = np.tile(self._dry_src_day, (n_steps, 1))
            data = np.where(dry, self.src_mean_vec.astype(data.dtype), data)
        if self._bathymetry_geoinfo:
            geoinfos = np.tile(self._geo_day, (n_steps, 1))
        else:
            geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(datetimes)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        return rd

    @override
    def get_target(self, idx: TIndex) -> ReaderData:
        self._lazy_init()
        (t_idxs, dtr) = self._get_target_idxs(idx)
        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(self.source_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = t_idxs[0] * self.target_mesh_size
        end_row = (t_idxs[-1] + 1) * self.target_mesh_size

        # Note: we read all columns from start_row to end_row once,
        # then select the ones we need. This is more efficient for Zarr.
        full_data_slice = self._read_slice(self.t_data_list, self.t_data_cum, start_row, end_row)
        datetimes_lazy = self._read_slice(self.t_times_list, self.t_times_cum, start_row, end_row)

        # Define the specific slices we need from the larger block
        data_lazy = full_data_slice[:, self.target_idx]

        # Dask optimizes this to a single (or few) efficient read operation(s).
        data, datetimes = dask.compute(
            data_lazy, datetimes_lazy, scheduler="single-threaded"
        )
        
        n_steps = data.shape[0] // self.target_mesh_size
        coords = np.tile(self.cached_target_coords, (n_steps, 1))
        if self._wet_mask_from_mesh:
            # dry targets are NaN so the loss skips them
            dry = np.tile(self._dry_trg_day, (n_steps, 1))
            data = np.where(dry, np.nan, data)
        if self._bathymetry_geoinfo:
            geoinfos = np.tile(self._geo_day, (n_steps, 1))
        else:
            geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(datetimes)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        return rd

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        return self.get_source(idx)

    @override
    def normalize_source_channels(self, source: NDArray) -> NDArray:
        """
        Normalize source channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        self._lazy_init()
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        # values marking no-data (e.g. below-bathymetry fill) are replaced by the channel mean,
        # i.e. they become 0 after normalisation. Setting them to 0 afterwards is the same but
        # exact for float32 data with float64 statistics.
        fill_value = self._stream_info.get("source_fill_value")
        no_data = None if fill_value is None else source == fill_value
        source -= self.src_mean_vec
        source /= self.src_stdev_vec
        if no_data is not None:
            source[no_data] = 0.0

        return source

    @override
    def normalize_target_channels(self, target: NDArray) -> NDArray:
        """
        Normalize target channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        self._lazy_init()
        assert target.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        # values marking no-data (e.g. below-bathymetry fill) become NaN so the loss
        # masks them; sources are left untouched since the encoder needs finite inputs
        fill_value = self._stream_info.get("target_fill_value")
        if fill_value is not None:
            target = np.where(target == fill_value, np.nan, target)
        target -= self.trg_mean_vec
        target /= self.trg_stdev_vec

        return target

    @override
    def denormalize_source_channels(self, source: NDArray) -> NDArray:
        """
        Denormalize source channels

        Parameters
        ----------
        data :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        self._lazy_init()
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        source *= self.src_stdev_vec
        source += self.src_mean_vec

        return source

    @override
    def denormalize_target_channels(self, data: NDArray) -> NDArray:
        """
        Denormalize target channels

        Parameters
        ----------
        data :
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized data
        """
        self._lazy_init()
        assert data.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        data *= self.trg_stdev_vec
        data += self.trg_mean_vec

        return data
