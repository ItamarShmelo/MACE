"""Reader for MACE Compton table cache files (mace_compton_<key>.tbl).

The file is a text header ending with ``END_HEADER`` followed by the arrays
named on the ``arrays`` line as raw little-endian doubles:
boundaries (G+1) [erg], centers (G) [erg], temperatures (NT) [K], then
sigma and dsigma_dT, each NT x G x G x A, indexed [T][g][g'][angle bin]
(per free electron, cm^2 and cm^2/K).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_table(path: str | Path) -> dict:
    """Return the header fields and the arrays of a Compton table file."""
    data = Path(path).read_bytes()
    marker = b"END_HEADER\n"
    end = data.index(marker)
    lines = data[:end].decode().splitlines()
    if not lines or lines[0].split()[:2] != ["MACE_COMPTON_TABLE", "1"]:
        raise ValueError(f"{path} is not a version-1 MACE Compton table")
    header: dict[str, list[str]] = {}
    for line in lines[1:]:
        fields = line.split()
        if fields:
            header[fields[0]] = fields[1:]
    groups = int(header["groups"][0])
    angles = int(header["angle_bins"][0])
    count = int(header["temperature_count"][0])
    values = np.frombuffer(data, dtype="<f8", offset=end + len(marker))
    sizes = {
        "boundaries": groups + 1,
        "centers": groups,
        "temperatures": count,
        "sigma": count * groups * groups * angles,
        "dsigma_dT": count * groups * groups * angles,
    }
    result: dict = {"header": header}
    offset = 0
    for name in header["arrays"]:
        size = sizes[name]
        array = values[offset : offset + size]
        if array.size != size:
            raise ValueError(f"{path} is truncated in {name}")
        if name in ("sigma", "dsigma_dT"):
            array = array.reshape(count, groups, groups, angles)
        result[name] = array
        offset += size
    result["key"] = header["key"][0]
    result["mace_commit"] = header["mace_commit"][0]
    return result
