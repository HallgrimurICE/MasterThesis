from typing import Tuple, Any

import numpy as np
import haiku as hk
import jax.numpy as jnp
from numpy.lib.npyio import NpzFile


def _maybe_unwrap(x: Any) -> Any:
    """Unwraps numpy object arrays that just hold a single Python object."""
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        # scalar object array -> return the underlying object
        try:
            return x.item()
        except Exception:
            return x
    return x


class NpzParameterProvider:
    """
    Loads and exposes network params saved in a variety of ways:

    1) Named NPZ:
       np.savez("sl_params.npz", params=params, state=net_state, step=step)

       -> np.load(...) returns an NpzFile with .files including 'params'.

    2) Raw tuple/list:
       np.save("sl_params.npz", (params, net_state, step))

       -> np.load(..., allow_pickle=True) returns a tuple (or list).

    3) Dict-like:
       {'params': ..., 'state': ..., 'step': ...}

    This class normalises all of these into:
       self._params    : hk.Params tree
       self._net_state : hk.Params tree
       self._step      : int
    """

    def __init__(self, path: str):
        loaded = np.load(path, allow_pickle=True)

        # Case 1: standard NPZ archive with named entries
        if isinstance(loaded, NpzFile):
            self._from_npzfile(loaded)

        # Case 2: tuple/list: (params, state, step)
        elif isinstance(loaded, (tuple, list)):
            self._from_sequence(loaded)

        # Case 3: dict-like object with keys
        elif isinstance(loaded, dict):
            self._from_dict(loaded)

        else:
            raise TypeError(
                f"Unrecognised checkpoint type from {path}: {type(loaded)}"
            )

    # ---------- Different input formats ----------

    def _from_npzfile(self, data: NpzFile) -> None:
        files = list(data.files)

        if "params" not in files:
            raise ValueError(
                f"Expected 'params' in NPZ, found {files}. "
                "If this is actually a tuple, remove it and re-save as (params, state, step)."
            )

        params = data["params"]
        state = (
            data["state"]
            if "state" in files
            else data["net_state"]
            if "net_state" in files
            else {}
        )
        step = data["step"] if "step" in files else 0

        self._params = _maybe_unwrap(params)
        self._net_state = _maybe_unwrap(state)
        self._step = int(np.array(step).item()) if step is not None else 0

    def _from_sequence(self, seq: Any) -> None:
        if len(seq) != 3:
            raise ValueError(
                f"Expected tuple/list of length 3 (params, state, step), got length {len(seq)}"
            )
        params, state, step = seq
        self._params = _maybe_unwrap(params)
        self._net_state = _maybe_unwrap(state)
        self._step = int(step) if step is not None else 0

    def _from_dict(self, d: dict) -> None:
        if "params" not in d:
            raise ValueError(f"Expected key 'params' in dict, got keys {list(d.keys())}")
        params = d["params"]
        state = d.get("state", d.get("net_state", {}))
        step = d.get("step", 0)

        self._params = _maybe_unwrap(params)
        self._net_state = _maybe_unwrap(state)
        self._step = int(step) if step is not None else 0

    # ---------- Public API used by SequenceNetworkHandler ----------

    def params_for_actor(self) -> Tuple[hk.Params, hk.Params, jnp.ndarray]:
        """
        Interface compatible with parameter_provider.ParameterProvider.
        Returns (params, net_state, step).
        """
        return self._params, self._net_state, jnp.asarray(self._step, dtype=jnp.int32)
