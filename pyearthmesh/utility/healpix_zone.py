"""Helpers to create readable HEALPix zone names.

Provides a simple, human-friendly naming scheme for HEALPix pixels.

Functions
- generate_healpix_zone_name: return a readable string for a given (nside, pix).

Notes
- Hierarchical labels are only supported for NESTED ordering (because RING
  ordering does not preserve simple parent/child integer relationships).
"""
import math
from typing import List


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def generate_healpix_zone_name(
    nside: int,
    pix: int,
    nest: bool = True,
    hierarchical: bool = False,
    prefix: str = "HPX",
    pad_pix: bool = True,
) -> str:
    """Generate a readable zone name for a HEALPix pixel.

    Parameters
    - nside: HEALPix NSIDE (must be a power of two)
    - pix: pixel index (0 <= pix < 12*nside**2)
    - nest: whether the pixel index uses NESTED ordering (default True)
    - hierarchical: when True and `nest` is True, include a hierarchical
      path from the 12 base faces down to the requested pixel (uses successive
      integer division by 4 to determine parents/children)
    - prefix: textual prefix for the label (default "HPX")
    - pad_pix: pad pixel number with zeros to a fixed width based on nside

    Returns a short, readable string such as:
      HPX_n4_f3-01-2_pix37_nest

    Hierarchical example (nside=4):
      HPX_n4_f3-c0c1c2_pix37_nest  (where c0..c2 are child indices 0..3)

    Raises ValueError for invalid inputs.
    """
    if not _is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    npix = 12 * (nside ** 2)
    if not (0 <= pix < npix):
        raise ValueError(f"pix must be in [0, {npix})")

    # Basic parts
    pad_width = len(str(npix - 1)) if pad_pix else 0
    pix_str = str(pix).zfill(pad_width) if pad_width else str(pix)

    if hierarchical and not nest:
        raise ValueError("hierarchical labels are only supported for nest=True")

    parts: List[str] = [f"{prefix}_n{nside}"]

    if hierarchical and nest:
        # Number of subdivision levels relative to nside=1
        levels = int(math.log2(nside)) if nside > 1 else 0
        # Top-level face index at nside=1
        face = pix // (4 ** levels) if levels > 0 else pix
        children: List[int] = []
        for j in range(levels, 0, -1):
            child = (pix // (4 ** (j - 1))) % 4
            children.append(int(child))

        parts.append(f"f{face}")
        if children:
            parts.append("c" + "".join(str(c) for c in children))
    else:
        # Non-hierarchical label: include pixel index
        parts.append(f"pix{pix_str}")

    # Suffix to indicate ordering
    order = "nest" if nest else "ring"

    label = "-".join(parts) + f"_{order}"
    # If hierarchical, also show the final pixel index for clarity
    if hierarchical and nest:
        label = "-".join(parts + [f"pix{pix_str}"]) + f"_{order}"

    return label


if __name__ == "__main__":
    # Small interactive demo when run directly
    examples = [
        (1, 5, True, True),
        (2, 5, True, True),
        (4, 37, True, True),
        (8, 123, True, True),
        (4, 37, False, False),
    ]
    for nside, pix, nest, hier in examples:
        print(generate_healpix_zone_name(nside, pix, nest=nest, hierarchical=hier))
