"""
SEDS9 — DS9 Interactive SED Tool
=================================
Real-time spectral energy distribution plotting from DS9 regions.

Usage:
    From the command line:
        seds9
        seds9 --flux-unit Jy --log --target kepler

    From Python:
        from seds9 import DS9XPA, SEDExtractor, SEDPlotter, load_frame_info
"""

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy imports — only load the heavy modules when accessed."""
    _public = {
        'DS9XPA', 'FrameInfo', 'ParsedRegion', 'SEDResult',
        'SEDExtractor', 'SEDPlotter', 'load_frame_info',
        'parse_all_ds9_regions', 'simplify_polygon', 'FILTER_WAVELENGTHS',
    }
    if name in _public:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module 'seds9' has no attribute {name!r}")
