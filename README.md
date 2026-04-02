# SEDS9 — DS9 Interactive SED Tool

Real-time spectral energy distribution plotting from DS9 regions. Load multi-filter images in separate DS9 frames, draw regions of any shape, and watch the SED update live as you move or reshape them.

Supports **circle, ellipse, polygon, and box** regions, including complex contour-derived polygons with hundreds of vertices. Draw multiple labeled regions to compare SEDs from different spatial structures simultaneously.

## Install

```bash
# From the package directory
pip install .

# Or with photutils for better circle/ellipse photometry
pip install ".[all]"

# Or directly from a git repo
pip install git+https://github.com/noah-zimmer/seds9.git
```

After install, the `seds9` command is available everywhere:

```bash
seds9 --help
```

## Quick Start

```bash
# 1. Open DS9, load one filter per frame, tile them
ds9 -title kepler kepler_f770w.fits kepler_f1000w.fits kepler_f1500w.fits &

# 2. Draw a region on any frame

# 3. Run
seds9 --target kepler --coord-system wcs --flux-unit Jy --no-bg
```

## Usage

```bash
# Basic — auto-detect everything
seds9

# JWST MIRI with flux conversion and log axes
seds9 --flux-unit Jy --log

# Target a specific DS9 instance
seds9 --target kepler

# No background subtraction (good for extended sources)
seds9 --no-bg --flux-unit Jy --log

# Save SED data to CSV on each update
seds9 --flux-unit Jy --log --save-csv kepler_sed.csv

# Pixel coordinates (frames must be aligned)
seds9 --coord-system image

# Generate a DS9 analysis menu file
seds9 --generate-ds9-file sed_tool.ds9
```

## Multi-Region Comparison

Draw multiple regions in DS9, label each one (double-click region → Properties → Text), and each gets its own SED curve:

```bash
# Draw polygons around different structures, label them:
#   "ejecta knot", "CSM filament", "Green Monster"
seds9 --flux-unit Jy --log --save-csv regions.csv
```

## DS9 Menu Integration

```bash
seds9 --generate-ds9-file sed_tool.ds9
```

Then in DS9: **Edit → Preferences → Analysis → Load** → select `sed_tool.ds9`.

## Requirements

- SAOImage DS9 with XPA enabled (default)
- Python ≥ 3.8
- astropy, matplotlib, numpy
- photutils (optional, improves circle/ellipse photometry)
- XPA command-line tools (`xpaget`/`xpaset`) on PATH
- A non-Tk matplotlib backend (PyQt5/6 recommended) if launching from DS9's Analysis menu

## Supported Filters

Built-in wavelength mappings for JWST MIRI, JWST NIRCam, Spitzer IRAC/MIPS, and Herschel PACS/SPIRE. Custom mappings via `--wavelength FILTER=WAVELENGTH`.
