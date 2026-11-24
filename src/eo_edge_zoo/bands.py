"""Sentinel-2 band names + reordering. Pure python (no torch)."""
from collections.abc import Sequence

# SSL4EO-S12 order (torchgeo _sentinel2_toa_bands) = input order of S2 SSL weights
S2_SSL_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
RGB = ("B04", "B03", "B02")


def band_index(src_names: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """Indices into src_names giving `wanted` order; missing band raises."""
    if missing := set(wanted) - set(src_names):
        raise ValueError(f"missing bands: {sorted(missing)}")
    return [list(src_names).index(band) for band in wanted]
