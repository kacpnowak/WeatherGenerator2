# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch

import zarr
import typing
from weathergen.utils.config import Config


def sanitize_stream_str( istr) :
    return istr.replace( ' ', '_').replace( '-', '_').replace(',','')


def read_validation( cf, epoch, base_path, instruments, forecast_steps, rank=0) :

    streams, columns, data = [], [], []

    fname = base_path + 'validation_epoch{:05d}_rank{:04d}.zarr'.format( epoch, rank)
    store = zarr.DirectoryStore( fname)
    ds = zarr.group( store=store)

    for ii, stream_info in enumerate( cf.streams) :
          
        n = stream_info['name']
        if len(instruments) :
            if not np.array( [r in n for r in instruments]).any() :
              continue

        streams += [ stream_info['name'] ]
        columns.append( ds[f'{sanitize_stream_str(n)}/0'].attrs['cols'])
        data += [ [] ]

        for fstep in forecast_steps :
          
            data[-1] += [ [] ]
            istr = sanitize_stream_str(n)
            
            data[-1][-1].append( ds[f'{istr}/{fstep}/sources'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/sources_coords'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/preds'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/targets'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/targets_coords'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/sources_lens'])
            data[-1][-1].append( ds[f'{istr}/{fstep}/targets_lens'])
            data[-1][-1].append( ~np.isnan( data[-1][-1][3]) )

            data[-1][-1].append( np.mean( data[-1][-1][2], axis=1) )
            data[-1][-1].append( np.std( data[-1][-1][2], axis=1) )

    return streams, columns, data

def is_first_write_access(
  rn: str, forecast_steps: int, data_root: zarr.Group
) -> bool:
    """Determine if data has been written to the store."""
    if rn in data_root.group_keys() :
        if f'{forecast_steps}' not in data_root[rn].group_keys() :
          return True
    else :
        return True

    return False

def initial_write(
    data_root: zarr.Group,
    subgroup_name: str,
    forecast_steps: int,
    stream_info: dict[str, typing.Any],
    cols,
    stream_idx: int,
    source_k: np.ndarray,
    source_lens_k: np.ndarray,
    preds_k: np.ndarray,
    targets_k: np.ndarray,
    targets_coords_k: np.ndarray,
    targets_lens_k: np.ndarray,
    jac: np.ndarray | None = None
):
    """
    Initial write to the output data store.

    Args:
        data_root : root group of the zarr hierarchy
        subgroup_name : name used to identify zarr subgroup of particular stream
        forecast_steps : number of forcast engine iterations
        stream_info : parsed stream config file
        cols : column names of columns in obs_datasets_norm
        stream_idx : index of stream used to select data
        source_k : source of shape (n_sources, n_timesteps, 6+source_len?)
        source_lens_k : 0d array of lenght of source_k
        preds_k : all predictions for stream index k
        targets_k : all targets for stream index k
        targets_coords_k : all target coordinates for stream index k
        targets_lens_k : 0d array of lenght of targets_k
        jac : jacobian (default None)
    """
    ds_source = data_root.require_group(f"{subgroup_name}/{forecast_steps}")

    # column names
    if stream_info["type"] in ['anemoi', 'regular', 'unstr']:
        cols_values = np.arange(2, len(cols[stream_idx]))
    elif "obs" == stream_info["type"]:
        cols_values = [col[:9] == "obsvalue_" for col in cols[stream_idx]]
    else:
        assert False, "Unsuppported stream type"

    ds_source.attrs["cols"] = np.array(cols[stream_idx])[cols_values].tolist()
    ds_source.create_dataset(
        "sources", data=source_k, chunks=(1024, *source_k.shape[1:])
    )
    ds_source.create_dataset("sources_lens", data=source_lens_k)
    ds_source.create_dataset("preds", data=preds_k, chunks=(1024, *preds_k.shape[1:]))
    ds_source.create_dataset(
        "targets", data=targets_k, chunks=(1024, *targets_k.shape[1:])
    )
    ds_source.create_dataset(
        "targets_coords",
        data=targets_coords_k,
        chunks=(1024, *targets_coords_k.shape[1:]),
    )
    ds_source.create_dataset("targets_lens", data=targets_lens_k)

    if jac is not None:
        ds_source.create_dataset("jacobian", data=jac[stream_idx])

def successive_write(
    data_root,
    forecast_steps,
    source_k,
    source_lens_k,
    preds_k,
    targets_k,
    targets_coords_k,
    targets_lens_k,
    subgroup_name
):
    """
    Succecive write to the output data store.

    Args:
        data_root : root group of the zarr hierarchy
        forecast_steps : number of forcast engine iterations
        source_k : source of shape (n_sources, n_timesteps, 6+source_len?)
        source_lens_k : 0d array of lenght of source_k
        preds_k : all predictions for stream index k
        targets_k : all targets for stream index k
        targets_coords_k : all target coordinates for stream index k
        targets_lens_k : 0d array of lenght of targets_k
        subgroup_name : name used to identify zarr subgroup of particular stream
    """
    subgroup_name = subgroup_name + f'/{forecast_steps}'
    data_root[f'{subgroup_name}/sources'].append( source_k)
    data_root[f'{subgroup_name}/sources_lens'].append( source_lens_k)
    data_root[f'{subgroup_name}/preds'].append( preds_k)
    data_root[f'{subgroup_name}/targets'].append( targets_k)
    data_root[f'{subgroup_name}/targets_coords'].append( targets_coords_k)
    data_root[f'{subgroup_name}/targets_lens'].append( targets_lens_k)


def get_store(base_path, epoch, rank, jac):
    """Construct DirectoryStore and access root group."""
    fname = base_path + "validation_epoch{:05d}_rank{:04d}".format(epoch, rank)
    fname += "" if jac is None else "_jac"
    fname += ".zarr"
    
    store = zarr.DirectoryStore(fname)
    
    return zarr.group( store=store), store

def get_data_buffers(
  sources, preds_all, targets_all, targets_coords_all, targets_lens, stream_idx
) -> tuple[np.ndarray]:
    """
    Gather buffers for from all devices and prepare them for output writing.

    Args:
        sources : sources of shape (n_sources, n_timesteps, 6+source_len?)
        preds_all : per stream predictions
        targets_all : per stream targets
        targets_coords_all : per stream coordinates for targets
        targets_lens : lengths of targets ??
        stream_idx : index of stream used to select stream data from buffers

    Returns:
        Numpy buffers for a the stream at position stream_idx.
    """
    source_k = sources[0][stream_idx].cpu().detach().numpy()
    source_lens_k = np.array([source_k.shape[0]])
    preds_k = torch.cat(preds_all[stream_idx], 1).transpose(1, 0).cpu().detach().numpy()
    targets_k = torch.cat(targets_all[stream_idx], 0).cpu().detach().numpy()
    targets_coords_k = (
        torch.cat(targets_coords_all[stream_idx], 0).cpu().detach().numpy()
    )
    targets_lens_k = np.array(targets_lens[stream_idx], dtype=np.int64)

    return source_k, source_lens_k, preds_k, targets_k, targets_coords_k, targets_lens_k

def is_empty(targets):
    """Check if nothing should have been forecasted."""
    return len(targets) == 0 or len(targets[0]) == 0

def write_validation(
    cf: Config,
    base_path: str,
    rank: int,
    epoch: int,
    cols: list[str],
    sources: list[np.ndarray],
    preds_all: list[list[typing.Any]],
    targets_all: list[list[typing.Any]],
    targets_coords_all : list[list[typing.Any]],
    targets_lens : list[list[typing.Any]],
    jac: list[np.ndarray] | None = None
):
    """
    Output one epoch data.

    Args:
        cf : global config object
        base_path : results directory for current run
        rank : MPI rank
        epoch : current epoch
        cols : column names of columns in obs_datasets_norm
        sources : vectorized input of shape (n_sources, n_timesteps, 6+source_len?) or empty ??
        preds_all : per stream predictions ??
        targets_all : per stream targets ??
        target_coords_all : per stream coordinates for targets ??
        targets_lens : lengths of targets ??
        jac : per stream jacobian (default None)
    """

    data_root, store = get_store(base_path, epoch, rank, jac)

    for stream_idx, stream_info in enumerate(cf.streams):
        # skip empty entries (e.g. no channels from the sources are used as targets)
        targets = targets_all[stream_idx]
        if (
            (not is_empty(targets)) and
            (stream_info["name"] in cf.analysis_streams_output)
        ):
            # TODO: this only saves the first batch
            source_k, source_lens_k, preds_k, targets_k, targets_coords_k, targets_lens_k = get_data_buffers(
                sources, preds_all, targets_all, targets_coords_all, targets_lens, stream_idx
            )

            forecast_steps: int = cf.forecast_steps # number of forecast engine iteration steps

            # if forecast_steps is iterable obtain number of forecast_steps as int
            forecast_steps = (
                forecast_steps if isinstance(forecast_steps, int) else (
                    forecast_steps[min( epoch, len(forecast_steps)-1)]
                )
            )

            # magic string replacement (name ???)
            subgroub_name = stream_info['name'].replace(' ', '_').replace('-', '_').replace(',', '')

            if is_first_write_access(data_root, subgroub_name, forecast_steps):
                initial_write(
                    data_root, subgroub_name, forecast_steps, stream_info, cols, stream_idx, source_k, source_lens_k, preds_k, targets_k, targets_coords_k, targets_lens_k, jac
                )
            else :
                successive_write(
                    data_root, forecast_steps, source_k, source_lens_k, preds_k, targets_k, targets_coords_k, targets_lens_k, subgroub_name
                )

    store.close()
