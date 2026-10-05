# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from pathlib import Path
from typing import override

import anemoi.datasets as anemoi_datasets
import numpy as np
from anemoi.datasets.data import MissingDateError
from anemoi.datasets.data.dataset import Dataset
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.common.config import timedelta_to_str
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage
from weathergen.utils.distributed import is_root

_logger = logging.getLogger(__name__)

# Time-dependent "forcing" geoinfo channels: these are not weather fields but deterministic
# functions of (lat, lon, datetime). anemoi-datasets bakes them into the zarr via earthkit's
# forcings source; we recompute them identically so target queries can be built for future
# forecast steps that lie beyond the dataset coverage. Verified bit-close to the zarr (~1e-7).
_FORCING_GEOINFO_CHANNELS = frozenset(
    ["insolation", "cos_local_time", "sin_local_time", "cos_julian_day", "sin_julian_day"]
)
_DAYS_PER_YEAR = 365.25


def compute_forcing_geoinfos(
    lat: NDArray, lon: NDArray, when: datetime.datetime
) -> dict[str, NDArray]:
    """
    Reproduce the earthkit/anemoi time-dependent geoinfo forcings for a grid at one datetime.

    Mirrors earthkit.data ForcingMaker (julian_day / local_time) and
    earthkit.meteo.solar.cos_solar_zenith_angle (insolation). lat/lon are in degrees with the
    same convention as the dataset; `when` is a tz-naive UTC datetime.
    """
    from earthkit.meteo.solar import cos_solar_zenith_angle

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    delta_year = when - datetime.datetime(when.year, 1, 1)
    jd = delta_year.days + delta_year.seconds / 86400.0
    ang = jd / _DAYS_PER_YEAR * 2.0 * np.pi

    delta_day = when - datetime.datetime(when.year, when.month, when.day)
    utc_hour = (delta_day.days + delta_day.seconds / 86400.0) * 24.0
    local = (lon / 360.0 * 24.0 + utc_hour) % 24.0
    local_rad = local / 24.0 * 2.0 * np.pi

    insolation = np.clip(cos_solar_zenith_angle(when, lat, lon), 0.0, None)

    return {
        "cos_julian_day": np.full_like(lat, np.cos(ang)),
        "sin_julian_day": np.full_like(lat, np.sin(ang)),
        "cos_local_time": np.cos(local_rad),
        "sin_local_time": np.sin(local_rad),
        "insolation": np.asarray(insolation, dtype=np.float64),
    }


class DataReaderAnemoi(DataReaderTimestep):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # use anemoi_config if it's defined; ignore filename in this case
        data_paths = stream_info.get("data_paths", [])
        anemoi_config = stream_info.get("anemoi_config")
        if anemoi_config:
            # convert OmegaConf DictConfig to a plain dict for anemoi.open_dataset.
            filename = OmegaConf.to_container(anemoi_config, resolve=True)
            # add additional data paths
            for path in data_paths:
                anemoi_datasets.add_dataset_path(path)
            # provide some visibility since we ignore filename
            if is_root():
                _logger.info("Ignoring filename and using anemoi_config option.")

        # open  dataset to peak that it is compatible with requested parameters
        ds0: Dataset = anemoi_datasets.open_dataset(filename)
        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds0.dates[-1] or tw_handler.t_end <= ds0.dates[0]:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        kwargs = {}
        if "frequency" in stream_info:
            frequency = timedelta_to_str(stream_info["frequency"])
            kwargs["frequency"] = frequency
        if "subsampling_rate" in stream_info:
            name = stream_info["name"]
            _logger.warning(
                f"subsampling_rate specified for anemoi dataset for stream {name}. "
                + "Use frequency instead."
            )
        ds: Dataset = anemoi_datasets.open_dataset(
            ds0, **kwargs, start=tw_handler.t_start, end=tw_handler.t_end
        )

        period = np.timedelta64(ds.frequency)
        data_start_time = ds.dates[0]
        data_end_time = ds.dates[-1]
        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)

        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # select/filter requested source channels
        if stream_info.get(str(stage) + "_source_channels") is None:
            self.source_idx = self.select_channels(ds, "source")
            self.source_channels = [ds.variables[i] for i in self.source_idx]
        else:
            self.source_channels = stream_info.get(str(stage) + "_source_channels")
            self.source_idx = [ds.variables.index(ch) for ch in self.source_channels]

        # select/filter requested target channels
        if stream_info.get(str(stage) + "_target_channels") is None:
            self.target_idx = self.select_channels(ds, "target")
            self.target_channels = [ds.variables[i] for i in self.target_idx]
        else:
            self.target_channels = stream_info.get(str(stage) + "_target_channels")
            self.target_idx = [ds.variables.index(ch) for ch in self.target_channels]

        # get target channel weights from stream config
        if stream_info.get("target_channel_weights") is None:
            self.target_channel_weights = self.parse_target_channel_weights()
        else:
            self.target_channel_weights = stream_info.get("target_channel_weights")

        # select/filter requested geoinfo channels (can be any variable, not just constant-in-time)
        if stream_info.get("geoinfo_channels") is None:
            self.geoinfo_idx = self.select_geoinfo_channels(ds)
            self.geoinfo_channels = [ds.variables[i] for i in self.geoinfo_idx]
        else:
            self.geoinfo_channels = stream_info.get("geoinfo_channels")
            self.geoinfo_idx = [ds.variables.index(ch) for ch in self.geoinfo_channels]

        # set geoinfo normalization statistics
        if len(self.geoinfo_idx) > 0:
            self.mean_geoinfo = ds.statistics["mean"][self.geoinfo_idx]
            self.stdev_geoinfo = ds.statistics["stdev"][self.geoinfo_idx]
        else:
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)

        self.ens_member = 0
        if stream_info.get("ensemble_member") is not None:
            self.ens_member = stream_info.get("ensemble_member")
            _logger.info(f"{stream_info['name']}: using ensemble member {self.ens_member}")

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

        # cache of the time-invariant geoinfo channels (z, lsm, ...), filled lazily from the
        # first available window and reused to build target queries for free-running forecast
        # steps that lie beyond the dataset coverage.
        self._static_geoinfo: NDArray[np.float32] | None = None

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        try:
            data = self.ds[didx_start:didx_end][:, :, self.ens_member].astype(np.float32)
        except MissingDateError as e:
            _logger.debug(f"Date not present in anemoi dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # coords-first representation and collapse multiple steps
        data = data.transpose([0, 2, 1]).reshape((data.shape[0] * data.shape[2], -1))

        # extract geoinfo channels (can be time-varying, so read from dataset)
        geoinfos = data[:, list(self.geoinfo_idx)]
        # extract channels
        data = data[:, list(channels_idx)]

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(self.ds.dates[didx_start:didx_end], len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def _ensure_static_geoinfo(self) -> None:
        """Cache the geoinfo row from the first available window (static channels are time-
        invariant; the time-varying channels are overwritten per query)."""
        if self._static_geoinfo is not None or self.ds is None:
            return
        raw = self.ds[0:1][:, :, 0].astype(np.float32)
        raw = raw.transpose([0, 2, 1]).reshape((raw.shape[0] * raw.shape[2], -1))
        self._static_geoinfo = raw[:, list(self.geoinfo_idx)]

    def get_target_query(self, when: datetime.datetime) -> ReaderData:
        """
        Build a full-grid target "query" for a forecast step beyond the dataset coverage.

        Free-running forecasting has no ground truth past the data, but the decoder still needs
        the prediction locations (the full grid) and their geoinfos. Coords come from the fixed
        grid, static geoinfos from the cached window, and the time-dependent geoinfos are
        computed analytically for `when` (see compute_forcing_geoinfos). Data values are unknown
        and returned as zeros (predictions are written; there is no target to compare against).
        """
        self._ensure_static_geoinfo()

        coords = np.stack([self.latitudes, self.longitudes], axis=-1).astype(np.float32)
        geoinfos = self._static_geoinfo.copy()

        forcings = compute_forcing_geoinfos(self.latitudes, self.longitudes, when)
        for channel, values in forcings.items():
            if channel in self.geoinfo_channels:
                geoinfos[:, self.geoinfo_channels.index(channel)] = values.astype(np.float32)

        data = np.zeros((coords.shape[0], len(self.target_idx)), dtype=np.float32)
        datetimes = np.repeat(np.datetime64(when), coords.shape[0])

        return ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
            is_forecast_query=True,
        )

    def select_channels(self, ds0: anemoi_datasets, ch_type: str) -> NDArray[np.int64]:
        """
        Select source or target channels

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes

        """

        channels = self.stream_info.get(ch_type)
        channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        chs_idx = np.sort(
            [
                ds0.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f == k for f in channels]).any() if channels is not None else True
                    )
                    and not np.array([f == k for f in channels_exclude]).any()
                )
            ]
        )

        return np.array(chs_idx, dtype=np.int64)

    def select_geoinfo_channels(self, ds0: anemoi_datasets) -> NDArray[np.int64]:
        """
        Select geoinfo channels (can be any variable, not just constant-in-time)

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels

        Returns
        -------
        NDArray of channel indices for geoinfo variables

        """

        geoinfo_channels = self.stream_info.get("geoinfo_channels", [])

        if len(geoinfo_channels) == 0:
            return np.array([], dtype=np.int64)

        # Select channels that match the geoinfo list (exact match required)
        chs_idx = np.sort(
            [ds0.name_to_index[k] for k in ds0.typed_variables.keys() if k in geoinfo_channels]
        )

        if len(chs_idx) == 0 and len(geoinfo_channels) > 0:
            stream_name = self.stream_info["name"]
            _logger.warning(
                f"No matching geoinfo channels found for {stream_name}. "
                f"Requested: {geoinfo_channels}"
            )

        return np.array(chs_idx, dtype=np.int64)


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
