# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    ReaderData
)

from weathergen.datasets.stream_data import StreamData

from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)

_logger = logging.getLogger(__name__)


class DataReaderWeatherGenerator :
   " Wrapper for Output of WG"
   
   def __init__(
	    self,
        preds,
        streams,
        streams_dataset_val,
        healpix_level,
        fstep
        ) -> None:

       self.preds = preds
       self.streams = streams
       self.streams_datasets = streams_dataset_val.streams_datasets
       self.healpix_level = healpix_level
       self.fstep = fstep
       self.tokenizer = streams_dataset_val.tokenizer
       self.time_window_handler = streams_dataset_val.time_window_handler
       self.stream_configs = streams_dataset_val.streams
                   
   def get(self):
       streams_data: list[StreamData] =  []
       batch = []
       for batch_idx,batch_stream in enumerate(self.streams):
           for stream_idx,stream in enumerate(batch_stream):
               stream_data = StreamData(0,0,self.healpix_level,self.healpix_level)
               ds = self.streams_datasets[batch_idx][stream_idx]
               forcing_idx = ds.forcing_idx_in_source
               source_target_channel_order = ds.source_target_channel_order
               data = ds.denormalize_target_channels(self.preds[batch_idx][stream_idx]).cpu()
               coords = stream.target_coords_raw[self.fstep].cpu()
               datetimes = stream.target_times_raw[self.fstep]
               geoinfos = torch.zeros((data.shape[0],0), dtype=data.dtype).cpu()
               rdata = ReaderData(
                    coords=coords,
                    geoinfos=geoinfos.numpy(),
                    data=data[:,source_target_channel_order],
                    datetimes=datetimes)
               
               source_raw = torch.from_numpy(
                    np.concatenate((rdata.coords, rdata.geoinfos, rdata.data), 1)
                )
                
               time_win1 = self.time_window_handler.window_from_time(
                       sorted(np.unique(datetimes)))
            
               (ss_cells, ss_lens, ss_centroids) = self.tokenizer.batchify_source(
                    self.stream_configs[stream_idx],
                    rdata.coords,
                    torch.from_numpy(rdata.geoinfos),
                    rdata.data,
                    rdata.datetimes,
                    (time_win1.start, time_win1.end),
                    ds,
                )
               ds.normalize_target_channels(self.preds[batch_idx][stream_idx])

               stream_data.add_source(source_raw, ss_lens, ss_cells, ss_centroids)
               merge_inputs(stream_data,12288)
               streams_data += [stream_data]
           batch += [streams_data]
       source_cell_lens = compute_source_cell_lens(batch)
       batch = compute_offsets_scatter_embed(batch)
       
       return batch, source_cell_lens


def merge_cells(s_list,num_healpix_cells):
    cat = torch.cat
    ret = []
    for i in range(num_healpix_cells):
        ret_cell = []
        for i_s in range(len(s_list)):
            ret_cell.append(s_list[i_s][i])
        ret_cell = cat(ret_cell,0)
        ret.append(ret_cell)
    ret = cat(ret)
    return ret

def merge_inputs(stream_data,num_healpix_cells):
    source_tokens_cells = stream_data.source_tokens_cells
    source_centroids = stream_data.source_centroids
    
    stream_data.source_tokens_cells = merge_cells(source_tokens_cells, num_healpix_cells)
    stream_data.source_centroids = merge_cells(source_centroids, num_healpix_cells)
    stream_data.source_tokens_lens = torch.stack(stream_data.source_tokens_lens).sum(0)
    stream_data.target_coords[0] = torch.tensor([])
    stream_data.target_coords_raw[0] = torch.tensor([])
    stream_data.target_times_raw[0] = np.array([], dtype="datetime64[ns]")
    stream_data.target_tokens[0] = torch.tensor([])
    stream_data.target_tokens_lens[0] = torch.tensor([0])
    stream_data.target_coords_lens[0] = torch.tensor([])



def convert_wg_output_to_dataloader_output(preds,streams_data, streams_dataset_val, healpix_level,fstep):
    """
     Processes the ouput of weathergenerator such that it can process back again; usecase rollut in physical space
     preds: list of ensemble predicted tensors [num_streams, ensemble_members, grid_points, predictions]
    """

    ds = DataReaderWeatherGenerator(preds,streams_data,streams_dataset_val,healpix_level, fstep)
    batch = ds.get()

    return [[d.to_device() for d in db] for db in batch[0]], batch[1].to("cuda"),

