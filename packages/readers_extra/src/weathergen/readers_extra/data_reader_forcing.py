# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi

_logger = logging.getLogger(__name__)


class DataReaderForcing(DataReaderAnemoi):
    """
    Per-step forcing reader (e.g. SST) backed by an anemoi dataset.

    A forcing stream is a prescribed field that is *not predicted* and is re-read at
    every forecast step's valid time, then injected into the forecasting engine
    (``entry_point: forecast_engine``). This differs from:

      * a normal ``anemoi`` source stream, which is only assimilated in the analysis
        window (t <= 0), and
      * a ``condition`` stream, which is a global scalar per step (no spatial field).

    All the actual reading (open_dataset, channel selection, normalisation, coord
    handling) is inherited unchanged from :class:`DataReaderAnemoi`; this subclass
    exists purely as a dedicated stream *type* so the sampler can route it through
    the per-step forcing path and the model can select an injection mode. The reader
    is constructed with the same signature as ``DataReaderAnemoi`` (tw_handler,
    filename, stream_info, stage).
    """

    # Marker consumed by the sampler to route this stream to the per-step forcing
    # path rather than the assimilation (t<=0 source) path.
    is_per_step_forcing: bool = True
