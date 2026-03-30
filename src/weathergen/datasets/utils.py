# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import astropy_healpix as hp
import numpy as np
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.batch import BatchSamples


####################################################################################################
def vecs_to_rots(vecs):
    """
    Convert vectors to rotations that align with (1,0,0) ie coordinate origin in geophysical
    spherical coordinates. A variant of Rodrigues formula is used.
    Additionally, applies a 2D rotation around the X-axis to align the local Z-axis
    with Global North, ensuring a consistent local tangent plane orientation.
    """
    device = vecs.device
    rots = torch.zeros((vecs.shape[0], 3, 3), dtype=torch.float64, device=device)
    c1 = vecs[:, 0].to(torch.float64)
    c2 = vecs[:, 1].to(torch.float64)
    c3 = vecs[:, 2].to(torch.float64)
    s = torch.square(c2) + torch.square(c3)
    
    # Prevent division by zero at exact poles/equator singularities
    s_safe = torch.where(s < 1e-12, torch.ones_like(s), s)
    
    # Original rotation mapping vecs to (1,0,0)
    rots[:, 0, 0] = c1
    rots[:, 0, 1] = c2
    rots[:, 0, 2] = c3
    rots[:, 1, 0] = -c2
    rots[:, 1, 1] = torch.where(s < 1e-12, torch.ones_like(c1), (c1 * torch.square(c2) + torch.square(c3)) / s_safe)
    rots[:, 1, 2] = torch.where(s < 1e-12, torch.zeros_like(c1), (-1.0 + c1) * c2 * c3 / s_safe)
    rots[:, 2, 0] = -c3
    rots[:, 2, 1] = torch.where(s < 1e-12, torch.zeros_like(c1), (-1.0 + c1) * c2 * c3 / s_safe)
    rots[:, 2, 2] = torch.where(s < 1e-12, torch.ones_like(c1), (torch.square(c2) + c1 * torch.square(c3)) / s_safe)

    # 1. Compute Global North vector n_global for each point
    nx = -c1 * c3
    ny = -c2 * c3
    nz = 1.0 - c3**2
    n_norm = torch.sqrt(nx**2 + ny**2 + nz**2)
    n_norm_safe = torch.where(n_norm < 1e-12, torch.ones_like(n_norm), n_norm)
    
    nx = nx / n_norm_safe
    ny = ny / n_norm_safe
    nz = nz / n_norm_safe
    
    # 2. Rotate Global North to the local frame
    n_local_x = rots[:, 0, 0]*nx + rots[:, 0, 1]*ny + rots[:, 0, 2]*nz
    n_local_y = rots[:, 1, 0]*nx + rots[:, 1, 1]*ny + rots[:, 1, 2]*nz
    n_local_z = rots[:, 2, 0]*nx + rots[:, 2, 1]*ny + rots[:, 2, 2]*nz

    # 3. Compute 2D rotation Rx around the local X-axis to map n_local to the local Z-axis (0,0,1)
    n_yz_norm = torch.sqrt(n_local_y**2 + n_local_z**2)
    n_yz_norm_safe = torch.where(n_yz_norm < 1e-12, torch.ones_like(n_yz_norm), n_yz_norm)
    
    sin_a = n_local_y / n_yz_norm_safe
    cos_a = n_local_z / n_yz_norm_safe
    
    Rx = torch.zeros_like(rots)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cos_a
    Rx[:, 1, 2] = -sin_a
    Rx[:, 2, 1] = sin_a
    Rx[:, 2, 2] = cos_a
    
    # Saftey bypass if the cell is exactly at the poles
    is_pole = (n_norm < 1e-12) | (n_yz_norm < 1e-12)
    Rx[is_pole, 1, 1] = 1.0
    Rx[is_pole, 1, 2] = 0.0
    Rx[is_pole, 2, 1] = 0.0
    Rx[is_pole, 2, 2] = 1.0
    
    # 4. Final rotation matrix combines both transformations
    rots_aligned = torch.bmm(Rx, rots)

    return rots_aligned


####################################################################################################
def s2tor3(lats, lons):
    """
    Convert from spherical to Cartesion R^3 coordinates

    Note: mathematics convention with lats in [0,pi] and lons in [0,2pi] is used
          (which is not problematic for lons but for lats care is required)
    """
    sin_lats = torch.sin(lats)
    cos_lats = torch.cos(lats)

    # Calculate the x, y, and z coordinates using vectorized operations.
    x = sin_lats * torch.cos(lons)
    y = sin_lats * torch.sin(lons)
    z = cos_lats

    # Stack the x, y, and z tensors along the last dimension.
    return torch.stack([x, y, z], dim=-1)


####################################################################################################
def r3tos2(pos: torch.Tensor) -> torch.Tensor:
    """
    Convert from spherical to Cartesion R^3 coordinates

    This optimized version is faster and more numerically stable by:
    1. Unbinding the input tensor to get x, y, and z components directly.
    2. Using torch.hypot for a more efficient and stable calculation of
       the xy-plane norm.
    3. Stacking the final latitude and longitude tensors along the last
       dimension, which avoids an expensive permute operation.

    Args:
        pos (torch.Tensor): A tensor of Cartesian coordinates with shape `(..., 3)`.

    Returns:
        torch.Tensor: .
    """
    # Unbind the last dimension to get x, y, and z tensors.
    x, y, z = torch.unbind(pos, dim=-1)

    # Use torch.hypot(x, y)
    xy_norm = torch.sqrt(x**2 + y**2)

    # Calculate latitudes and longitudes using atan2.
    # The output is directly a tensor with the same batch dimensions as the input.
    lats = torch.atan2(z, xy_norm)
    lons = torch.atan2(y, x)

    # Stack the results along the final dimension to get a `(..., 2)` tensor.
    return torch.stack([lats, lons], dim=-1)


####################################################################################################
def locs_to_cell_coords(hl: int, locs: list, dx=0.5, dy=0.5) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    assert locs[13].shape[-1] == 3 if len(locs[13]) > 0 else True

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    assert len(locs) == num_healpix_cells

    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    healpix_centers = s2tor3(
        torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value)
    )
    healpix_centers_rots = vecs_to_rots(healpix_centers)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin
    local_locs = [
        torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1) if len(s) > 0 else torch.tensor([])
        for i, (R, s) in enumerate(zip(healpix_centers_rots, locs, strict=False))
    ]

    return local_locs


####################################################################################################
def locs_to_ctr_coords(ctrs_r3, locs: list[torch.Tensor]) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    ctrs_rots = vecs_to_rots(ctrs_r3).to(torch.float32)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin

    # Concatenate all points into single tensor
    all_points = torch.cat(locs, dim=0)

    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )

    point_rotations = ctrs_rots[batch_indices]

    # Single vectorized batch matrix multiplication
    rotated_points = torch.bmm(point_rotations, all_points.unsqueeze(-1)).squeeze(-1)

    # Split back using tensor operations
    local_locs = torch.split(rotated_points, lengths.tolist())

    return list(local_locs)


####################################################################################################
def healpix_verts(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))

    return verts


####################################################################################################
def healpix_verts_rots(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))
    verts_rot3 = vecs_to_rots(verts)

    return verts, verts_rot3


####################################################################################################
def locs_to_cell_coords_ctrs(
    healpix_centers_rots: torch.Tensor, locs: list[torch.Tensor]
) -> torch.Tensor:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin

    # Concatenate all non-empty locations
    all_points = torch.cat(locs, dim=0)
    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)

    # Efficiently create batch indices using torch.repeat_interleave
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )

    # Select rotation matrices for each point
    rotations_selected = healpix_centers_rots[batch_indices]

    # Vectorized matrix multiplication
    local_locs = torch.bmm(rotations_selected, all_points.unsqueeze(-1)).squeeze(-1)

    return local_locs


####################################################################################################
def coords_to_hpyidxs(hl, thetas, phis):
    thetas = ((90.0 - thetas) / 180.0) * np.pi
    phis = ((180.0 + phis) / 360.0) * 2.0 * np.pi
    hpyidxs = ang2pix(2**hl, thetas, phis, nest=True)

    return hpyidxs


####################################################################################################
def add_local_vert_coords(hl, a, verts, tcs, zi, dx, dy, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = locs_to_cell_coords(hl, verts.unsqueeze(1), dx, dy)
    aa = ref - torch.cat(
        [
            aaa.repeat([*tt.shape[:-1], 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs, aa, strict=False)
        ]
    )
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = aa
    return a


####################################################################################################
def add_local_vert_coords_ctrs2(verts_local, tcs_lens, a, zi, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = ref - torch.cat(
        [
            aaa.unsqueeze(0).repeat([*tcs_lens, 1, 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs_lens, verts_local, strict=False)
        ],
        0,
    )
    aa = aa.flatten(1, 2)
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + aa.shape[-1])] = aa
    return a


def get_tokens_lens(streams: dict, batch_data: BatchSamples, input_steps: int) -> torch.Tensor:
    """
    Extract tokens_lens for (num_steps, num_samples, num_streams)
    """
    # collect source_tokens_lens for all stream datas
    source_tokens_lens = torch.stack(
        [
            torch.stack(
                [
                    torch.stack(
                        [
                            sample.streams_data[stream_info["name"]].source_tokens_lens[i]
                            for stream_info in streams
                        ]
                    )
                    for sample in batch_data.samples
                ]
                if len(batch_data.samples)
                else [torch.zeros((0, 0))]
            )
            for i in range(input_steps)
        ]
    )

    return source_tokens_lens
