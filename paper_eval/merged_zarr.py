# ── Open all per-rank validation archives of a run as ONE zarr store ──────────────────
# Nothing is copied: reads are routed to whichever archive owns the sample.
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph
from zarr.abc.store import Store
from zarr.core.array import AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import StorePath, ZipStore


class MergedRankStore(Store):
    """Read-only zarr store presenting several per-rank archives as one hierarchy.

    Each archive numbers its samples locally from 0, so they collide. Keys
    `<global_sample>/<rest>` are rewritten to `<local_sample>/<rest>` and
    forwarded to the archive that owns that sample.
    """

    supports_writes = supports_deletes = False
    supports_listing = True

    def __init__(self, stores, sample_map, paths, init_times):
        super().__init__(read_only=True)
        self._stores, self._paths = list(stores), [Path(p) for p in paths]
        self._sample_map = {str(g): (i, str(loc)) for g, (i, loc) in sample_map.items()}
        self.samples = sorted(sample_map)
        self.init_times = dict(init_times)
        self._is_open = True  # substores are already open
        self._root = json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"n_samples": len(self.samples)}}
        ).encode()

    def origin(self, sample):
        """`(archive, local sample)` backing a global sample."""
        i, local = self._sample_map[str(sample)]
        return self._paths[i], int(local)

    def _route(self, key):
        head, sep, rest = key.partition("/")
        entry = self._sample_map.get(head)
        if entry is None:
            return None
        i, local = entry
        return self._stores[i], (f"{local}/{rest}" if sep else local)

    async def get(self, key, prototype, byte_range=None):
        if key == "zarr.json":  # only ever fetched whole
            return prototype.buffer.from_bytes(self._root)
        routed = self._route(key)
        return None if routed is None else await routed[0].get(routed[1], prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        return [await self.get(k, prototype, r) for k, r in key_ranges]

    async def exists(self, key):
        routed = self._route(key)
        return True if key == "zarr.json" else routed is not None and await routed[0].exists(routed[1])

    async def getsize(self, key):
        routed = self._route(key)
        return len(self._root) if key == "zarr.json" else await routed[0].getsize(routed[1])

    async def list(self):
        yield "zarr.json"
        for s in self.samples:
            async for k in self.list_prefix(f"{s}/"):
                yield k

    async def list_prefix(self, prefix):
        prefix = prefix.lstrip("/")
        if prefix == "":
            async for k in self.list():
                yield k
            return
        routed = self._route(prefix)
        if routed is None:
            return
        head, (store, local_prefix) = prefix.partition("/")[0], routed
        local_head = local_prefix.partition("/")[0]
        async for k in store.list_prefix(local_prefix):
            yield f"{head}{k[len(local_head):]}"  # swap local sample id back to global

    async def list_dir(self, prefix):
        prefix = prefix.strip("/")
        if prefix == "":
            yield "zarr.json"
            for s in self.samples:
                yield str(s)
            return
        routed = self._route(prefix)
        if routed is None:
            return
        async for k in routed[0].list_dir(routed[1]):
            yield k

    def __eq__(self, other):
        return isinstance(other, MergedRankStore) and other._paths == self._paths

    def with_read_only(self, read_only=False):
        if read_only:
            return self
        raise NotImplementedError("read-only store")

    def close(self):
        for s in self._stores:
            s.close()
        super().close()

    async def set(self, key, value):
        raise NotImplementedError("read-only store")

    async def delete(self, key):
        raise NotImplementedError("read-only store")

    def __repr__(self):
        return f"MergedRankStore({len(self._stores)} archives, {len(self.samples)} samples)"


def _bytes(store, key):
    buf = sync(store.get(key, default_buffer_prototype()))
    return None if buf is None else buf.to_bytes()


def _children(store, prefix):
    names = sync(_alist(store.list_dir(prefix)))
    return [n for n in names if n != "zarr.json"]


async def _alist(it):
    return [x async for x in it]


def open_merged_ranks(results_dir, mini_epoch=0, order="time", ext="zip"):
    """Open every rank archive of a run as one zarr group.

    order="time": global sample i is the i-th forecast initialisation (samples are
    NOT stored in time order inside an archive). order="rank": concatenate archives
    in rank order, keeping each one's local ordering.
    """
    files = sorted(Path(results_dir).glob(f"validation_chkpt{mini_epoch:05d}_rank*.{ext}"))
    if not files:
        raise FileNotFoundError(f"no validation_chkpt{mini_epoch:05d}_rank*.{ext} in {results_dir}")

    stores = []
    for f in files:
        st = ZipStore(f, mode="r") if ext == "zip" else zarr.storage.LocalStore(f)
        sync(st._ensure_open())  # zip archives open lazily
        stores.append(st)

    entries, init_times = [], {}
    for i, st in enumerate(stores):
        local_samples = sorted(int(n) for n in _children(st, "") if n.isdigit())
        # Every dataset group carries a source_interval attribute and the hierarchy is
        # uniform, so probe one <stream>/<fstep>/<kind> path and reuse it for all samples.
        s0 = local_samples[0]
        stream = _children(st, str(s0))[0]
        fstep = _children(st, f"{s0}/{stream}")[0]
        kind = _children(st, f"{s0}/{stream}/{fstep}")[0]
        for loc in local_samples:
            meta = json.loads(_bytes(st, f"{loc}/{stream}/{fstep}/{kind}/zarr.json"))
            init_times[(i, loc)] = np.datetime64(meta["attributes"]["source_interval"]["start"])
            entries.append((i, loc))

    if order == "time":
        entries.sort(key=lambda e: (init_times[e], e))
    sample_map = dict(enumerate(entries))
    store = MergedRankStore(
        stores, sample_map, files, {g: init_times[e] for g, e in sample_map.items()}
    )
    return zarr.open_group(store=store, mode="r")


# ── Lazy xarray view of one stream ────────────────────────────────────────────────────
def _read_block(array, selection):
    return np.asarray(array[selection])[np.newaxis, np.newaxis]


def _stack_leaves(grid, name):
    """Assemble grid[sample][fstep] into one dask array, dims (sample, fstep, *leaf).

    The graph is emitted by hand: da.stack over tens of thousands of leaves spends
    minutes merging high-level graph layers. Chunking follows the SHARD shape --
    .chunks reports the inner chunk, and reading at that granularity would decode
    the enclosing shard once per inner chunk.
    """
    t = grid[0][0]
    leaf_chunks = normalize_chunks(
        t.shards if t.shards is not None else t.chunks, shape=t.shape, dtype=t.dtype
    )
    per_dim = []
    for dim_chunks in leaf_chunks:  # -> [(block index, slice), ...] for each dimension
        bounds = np.cumsum((0, *dim_chunks))
        per_dim.append([(i, slice(int(bounds[i]), int(bounds[i + 1]))) for i in range(len(dim_chunks))])
    blocks = [
        (tuple(i for i, _ in combo), tuple(s for _, s in combo))
        for combo in itertools.product(*per_dim)
    ]

    dsk = {
        (name, si, fi, *bidx): (_read_block, leaf, sel)
        for si, row in enumerate(grid)
        for fi, leaf in enumerate(row)
        for bidx, sel in blocks
    }
    chunks = ((1,) * len(grid), (1,) * len(grid[0]), *leaf_chunks)
    return da.Array(
        HighLevelGraph.from_collections(name, dsk, dependencies=[]), name, chunks, dtype=t.dtype
    )


def to_xarray(root, stream, kinds=("target", "prediction"), samples=None, fsteps=None,
              validate=64, max_workers=8):
    """Lazy dask-backed dataset for one stream: (sample, forecast_step, ipoint, channel[, ens]).

    validate: how many leaves to check against the template metadata (True = all,
    False = none); escalates to an exhaustive check if any spot-check disagrees.
    """
    kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    store = root.store
    samples = list(store.samples) if samples is None else [int(s) for s in samples]

    data_vars, coords = {}, {}
    for kind in kinds:
        avail = sorted(
            int(f) for f, g in root[f"{samples[0]}/{stream}"].groups() if kind in set(g.group_keys())
        )
        steps = avail if fsteps is None else [f for f in fsteps if f in set(avail)]
        if not steps:
            raise ValueError(f"no forecast step carries '{kind}' for stream {stream!r}")

        template = root[f"{samples[0]}/{stream}/{steps[0]}/{kind}"]
        if template["data"].shape[0] == 0:
            raise ValueError(f"{stream!r} has an empty '{kind}' (forcing stream?); try kinds='source'")

        paths = [f"{s}/{stream}/{f}/{kind}/data" for s in samples for f in steps]
        # Opening every leaf through zarr costs ms each; they share one metadata
        # document, so clone the template against each store path instead (~1000x).
        tmpl_arr, ref = root[paths[0]], _bytes(store, f"{paths[0]}/zarr.json") if validate else None

        def deviating(check):
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                raw = pool.map(lambda p: _bytes(store, f"{p}/zarr.json"), check)
                return {p for p, d in zip(check, raw, strict=True) if d != ref}

        bad = set()
        if validate is True:
            bad = deviating(paths)
        elif validate and deviating(paths[:: max(1, len(paths) // int(validate))]):
            bad = deviating(paths)

        opened = [
            root[p] if p in bad
            else zarr.Array(AsyncArray(metadata=tmpl_arr.metadata, store_path=StorePath(store, p)))
            for p in paths
        ]
        grid = [opened[i * len(steps):(i + 1) * len(steps)] for i in range(len(samples))]
        stacked = _stack_leaves(grid, f"wg-{stream}-{kind}-{len(paths)}")

        dims = ["sample", "forecast_step", "ipoint", "channel"] + (["ens"] * (stacked.ndim == 5))
        data_vars[kind] = xr.DataArray(
            stacked, dims=dims,
            coords={"forecast_step": steps, "channel": list(template.attrs["channels"])},
        )
        if "lat" not in coords:  # coords are identical for every sample/step of a stream
            pts = np.asarray(template["coords"])
            coords |= {"lat": ("ipoint", pts[:, 0]), "lon": ("ipoint", pts[:, 1])}

    ds = xr.Dataset(data_vars, coords=coords).assign_coords(
        sample=samples,
        init_time=("sample", np.array([store.init_times[s] for s in samples], "datetime64[ns]")),
        local_sample=("sample", [store.origin(s)[1] for s in samples]),
    )
    ds.attrs["stream"] = stream
    return ds


def add_valid_time(ds, root, stream):
    """Attach valid_time (sample, forecast_step). Every point of a sample/step shares
    one timestamp and the offset from init_time is constant, so read it once."""
    s0, kind = int(ds["sample"].values[0]), next(iter(ds.data_vars))
    init0 = np.datetime64(ds["init_time"].values[0])
    offs = np.array([
        np.datetime64(root[f"{s0}/{stream}/{int(f)}/{kind}/times"][0]) - init0
        for f in ds["forecast_step"].values
    ])
    valid = ds["init_time"].values[:, None] + offs[None, :]
    return ds.assign_coords(valid_time=(("sample", "forecast_step"), valid.astype("datetime64[ns]")))
