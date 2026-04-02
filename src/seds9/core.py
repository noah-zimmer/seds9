#!/usr/bin/env python3
"""
DS9 Interactive SED Tool
========================
Real-time spectral energy distribution plotting from DS9 regions.

Load multi-filter images of the same object in separate DS9 frames,
draw one or more regions, and this tool plots the SED — updating live
as you move, resize, or reshape the regions.

Supported region shapes: circle, ellipse, polygon, box

Multi-region mode (default):
  Draw multiple regions with DS9 text tags (double-click region →
  Properties → Text) to label different spatial structures (e.g.,
  "ejecta knot", "CSM shell", "Green Monster"). Each tagged region
  gets its own SED curve.

Requirements:
    - SAOImage DS9 with XPA enabled (default)
    - Python 3.8+
    - astropy, matplotlib, numpy
    - photutils (recommended for circle/ellipse apertures)
    - xpa command-line tools (xpaget/xpaset) on PATH

Usage:
    1. Open DS9 and load one filter per frame
    2. Draw regions (any shape) on any frame; tag them with names
    3. Run:  python ds9_sed_tool.py
    4. Move/resize regions — the SED plot updates in real time

Author: Generated for interactive JWST MIRI analysis
"""

import subprocess
import sys
import re
import os
import argparse
import warnings
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib

# Choose matplotlib backend carefully:
# - DS9 is a Tk application. If this script is spawned by DS9, using TkAgg
#   creates a second Tk event loop that deadlocks with DS9's.
# - QtAgg (PyQt5/6) or GTK3Agg/GTK4Agg are fully independent and safe.
# - TkAgg is fine when running from a standalone terminal.
# - The --backend flag lets the user override explicitly.
def _pick_backend():
    """Select a matplotlib backend, preferring Qt over Tk."""
    # Check if user specified --backend on the command line
    # (must happen before argparse, since imports happen at module load)
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            try:
                matplotlib.use(backend)
                return backend
            except Exception:
                pass  # fall through to auto-detect

    # Auto-detect: prefer non-Tk backends to avoid conflict with DS9
    for backend in ['QtAgg', 'Qt5Agg', 'GTK3Agg', 'GTK4Agg']:
        try:
            matplotlib.use(backend)
            return backend
        except ImportError:
            continue
    # Last resort: Tk (works from terminal, will conflict if DS9 spawns us)
    matplotlib.use('TkAgg')
    return 'TkAgg'

_selected_backend = _pick_backend()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
from matplotlib.path import Path as MplPath

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

try:
    from photutils.aperture import (
        CircularAperture, EllipticalAperture,
        ApertureStats,
    )
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False
    warnings.warn(
        "photutils not installed — falling back to mask-based sums. "
        "Install with: pip install photutils"
    )


# =============================================================================
# Filter Wavelength Database (microns)
# =============================================================================

FILTER_WAVELENGTHS = {
    # JWST MIRI imaging
    'F560W': 5.6,   'F770W': 7.7,   'F1000W': 10.0,  'F1130W': 11.3,
    'F1280W': 12.8,  'F1500W': 15.0,  'F1800W': 18.0,  'F2100W': 21.0,
    'F2550W': 25.5,
    # JWST MIRI coronagraphic
    'F1065C': 10.65, 'F1140C': 11.4,  'F1550C': 15.5,  'F2300C': 23.0,
    # JWST NIRCam short
    'F070W': 0.704,  'F090W': 0.901,  'F115W': 1.154,  'F150W': 1.501,
    'F200W': 1.989,  'F140M': 1.404,  'F162M': 1.627,  'F182M': 1.845,
    'F210M': 2.093,
    # JWST NIRCam long
    'F277W': 2.762,  'F356W': 3.568,  'F444W': 4.408,  'F300M': 2.989,
    'F335M': 3.362,  'F360M': 3.624,  'F410M': 4.082,  'F430M': 4.280,
    'F460M': 4.630,  'F480M': 4.874,
    # Spitzer IRAC
    'IRAC1': 3.6, 'IRAC2': 4.5, 'IRAC3': 5.8, 'IRAC4': 8.0,
    'I1': 3.6, 'I2': 4.5, 'I3': 5.8, 'I4': 8.0,
    # Spitzer MIPS
    'MIPS24': 24.0, 'MIPS70': 70.0, 'MIPS160': 160.0,
    # Herschel PACS
    'PACS70': 70.0, 'PACS100': 100.0, 'PACS160': 160.0,
    # Herschel SPIRE
    'SPIRE250': 250.0, 'SPIRE350': 350.0, 'SPIRE500': 500.0,
}

FILTER_HEADER_KEYS = ['FILTER', 'FILTER1', 'FILTER2', 'FILTNAM', 'FILTNAME',
                      'BAND', 'FILTID']

BUNIT_HEADER_KEYS = ['BUNIT', 'BUNITS']


# =============================================================================
# DS9 XPA Interface
# =============================================================================

class DS9XPA:
    """Communicate with DS9 via the XPA messaging system.

    Handles multiple DS9 instances gracefully by detecting XPA$BEGIN/END
    multi-response blocks and locking to a specific instance.
    """

    def __init__(self, target='ds9'):
        self.target = target
        self._resolve_target()
        self._verify_connection()

    def _resolve_target(self):
        """If multiple DS9 instances exist, pick one and lock to it.

        Selects the last (most recently registered) instance by default,
        since that's most likely the one you just launched. For explicit
        control, either:
          - Launch DS9 with: ds9 -title myname
            Then use:        --target myname
          - Or use the full XPA address shown in the instance list
        """
        try:
            result = subprocess.run(
                ['xpaget', self.target, 'version'],
                capture_output=True, text=True, timeout=10
            )
            raw = result.stdout.strip()
        except Exception:
            return  # let _verify_connection handle the failure

        # Check for multi-instance response
        if 'XPA$BEGIN' not in raw:
            return  # single instance, target is fine

        # Parse all instances: extract their XPA addresses
        # Format: XPA$BEGIN DS9:ds9 7f000001:33379\nds9 8.3\nXPA$END ...
        instances = re.findall(
            r'XPA\$BEGIN\s+(\S+\s+\S+)\n(.*?)\nXPA\$END',
            raw, re.DOTALL
        )

        if not instances:
            return

        if len(instances) == 1:
            self.target = instances[0][0]
            return

        print(f"  Found {len(instances)} DS9 instances:")
        for i, (addr, _) in enumerate(instances):
            print(f"    [{i+1}] {addr}")

        # Try interactive selection if stdin is a terminal
        chosen = None
        if sys.stdin.isatty():
            try:
                choice = input(f"  Select instance [1-{len(instances)}] "
                               f"(default: {len(instances)}, most recent): ").strip()
                if choice:
                    idx = int(choice) - 1
                    if 0 <= idx < len(instances):
                        chosen = instances[idx][0]
            except (ValueError, EOFError, KeyboardInterrupt):
                pass

        # Default: last instance (most recently launched)
        if chosen is None:
            chosen = instances[-1][0]

        self.target = chosen
        print(f"  Using: {chosen}")
        print(f"  Tip: launch DS9 with 'ds9 -title myname' and use "
              f"'--target myname' to skip this prompt")

    def _verify_connection(self):
        try:
            ver = self.get('version')
            print(f"  Connected to DS9 {ver}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to DS9 via XPA (target='{self.target}'). "
                "Ensure DS9 is running and xpaget/xpaset are on your PATH.\n"
                f"Error: {e}"
            )

    @staticmethod
    def _parse_xpa_response(raw: str) -> str:
        """Extract the payload from an XPA response.

        If the response contains XPA$BEGIN/END blocks (multiple instances
        or verbose mode), extract just the content from the first block.
        Otherwise return the raw string.
        """
        if 'XPA$BEGIN' not in raw:
            return raw.strip()

        # Extract content between first XPA$BEGIN and XPA$END
        m = re.search(r'XPA\$BEGIN\s+\S+\s+\S+\n(.*?)\nXPA\$END', raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        return raw.strip()

    def get(self, cmd: str) -> str:
        """Run xpaget and return stdout."""
        args = ['xpaget', self.target] + cmd.split()
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 and result.stderr.strip():
            raise RuntimeError(f"xpaget '{cmd}' failed: {result.stderr.strip()}")
        return self._parse_xpa_response(result.stdout)

    def set(self, cmd: str, data: Optional[str] = None):
        """Run xpaset."""
        args = ['xpaset', '-p', self.target] + cmd.split()
        if data is not None:
            args = ['xpaset', self.target] + cmd.split()
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, text=True)
            proc.communicate(input=data)
        else:
            subprocess.run(args, capture_output=True, timeout=10)

    def get_current_frame(self) -> int:
        return int(self.get('frame'))

    def set_frame(self, n: int):
        self.set(f'frame {n}')

    def get_filename(self) -> str:
        """Get FITS filename for the current frame.

        DS9 often returns paths with HDU bracket notation appended,
        e.g. '/path/to/file.fits[SCI,1]'. We strip that here so
        the result is a valid filesystem path, but preserve the
        original for informational purposes.
        """
        raw = self.get('file')
        # Strip DS9's HDU extension selector: file.fits[SCI] → file.fits
        if '[' in raw:
            raw = raw[:raw.index('[')]
        return raw

    def get_regions(self, coord_system: str = 'wcs', sky_frame: str = 'fk5') -> str:
        """Get region definitions from DS9."""
        if coord_system == 'wcs':
            cmd = f'regions -format ds9 -system wcs -sky {sky_frame}'
        else:
            cmd = f'regions -format ds9 -system {coord_system}'
        try:
            return self.get(cmd)
        except RuntimeError:
            return ''

    def discover_frames(self) -> List[int]:
        """Find all frames that have images loaded.

        Uses 'xpaget ds9 frame all' to get existing frame numbers
        without creating new ones, then checks which have files.
        """
        original = self.get_current_frame()

        # Get list of all existing frame numbers
        try:
            frame_str = self.get('frame all')
            all_frames = [int(x) for x in frame_str.split() if x.strip()]
        except (RuntimeError, ValueError):
            all_frames = [original]

        # Check which frames actually have files loaded
        frames = []
        for i in all_frames:
            try:
                self.set_frame(i)
                fname = self.get_filename()
                if fname:
                    frames.append(i)
            except (RuntimeError, subprocess.TimeoutExpired):
                continue

        self.set_frame(original)
        return frames if frames else [original]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FrameInfo:
    """Cached information about a single DS9 frame."""
    frame_num: int
    filename: str
    filter_name: str = 'UNKNOWN'
    wavelength_um: float = 0.0
    data: Optional[np.ndarray] = None
    header: Optional[fits.Header] = None
    wcs: Optional[WCS] = None
    bunit: str = ''
    pixar_sr: float = 0.0
    label: str = ''

    def __post_init__(self):
        if not self.label:
            self.label = self.filter_name


@dataclass
class ParsedRegion:
    """A single parsed DS9 region with metadata."""
    shape: str              # 'circle', 'ellipse', 'polygon', 'box'
    coord_system: str       # 'wcs' or 'image'
    raw_params: list        # raw coordinate/size strings
    label: str = ''         # from DS9 text={} or tag={}
    color: str = ''         # from DS9 color=
    index: int = 0          # order in region list


# =============================================================================
# Region Parsing — supports circle, ellipse, polygon, box, multi-region
# =============================================================================

def parse_all_ds9_regions(region_text: str, coord_system: str = 'wcs'
                          ) -> List[ParsedRegion]:
    """
    Parse ALL supported regions from DS9 region text.

    Extracts region shape, parameters, and metadata (text label, color, tags).
    Supports: circle, ellipse, polygon, box.

    DS9 region format examples:
        fk5
        circle(17:30:42.0,-21:29:28.0,10")  # text={ejecta knot} color=red
        polygon(17:30:40.0,-21:29:20.0,...) # text={CSM shell} color=green
        box(17:30:41.0,-21:29:30.0,20",15",45) # text={background}
    """
    lines = region_text.strip().split('\n')
    regions = []
    idx = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('global'):
            continue

        # Skip coordinate system declarations
        if line.lower() in ('fk5', 'icrs', 'galactic', 'ecliptic', 'image',
                            'physical', 'detector', 'amplifier', 'wcs'):
            continue

        # Skip excluded regions (prefixed with -)
        if line.startswith('-'):
            continue

        # Extract metadata from comments:  # text={label} color=red tag={foo}
        label = ''
        color = ''

        # text={...}
        m_text = re.search(r'text\s*=\s*\{([^}]*)\}', line, re.IGNORECASE)
        if m_text:
            label = m_text.group(1).strip()

        # tag={...}  — fallback label if no text
        if not label:
            m_tag = re.search(r'tag\s*=\s*\{([^}]*)\}', line, re.IGNORECASE)
            if m_tag:
                label = m_tag.group(1).strip()

        # color=
        m_color = re.search(r'color\s*=\s*(\w+)', line, re.IGNORECASE)
        if m_color:
            color = m_color.group(1).strip()

        # ---- Match shapes ----

        # circle(x,y,r)
        m = re.match(r'(?:-\s*)?circle\(([^)]+)\)', line, re.IGNORECASE)
        if m:
            parts = [p.strip() for p in m.group(1).split(',')]
            regions.append(ParsedRegion(
                shape='circle', coord_system=coord_system,
                raw_params=parts, label=label, color=color, index=idx
            ))
            idx += 1
            continue

        # ellipse(x,y,a,b,theta)
        m = re.match(r'(?:-\s*)?ellipse\(([^)]+)\)', line, re.IGNORECASE)
        if m:
            parts = [p.strip() for p in m.group(1).split(',')]
            regions.append(ParsedRegion(
                shape='ellipse', coord_system=coord_system,
                raw_params=parts, label=label, color=color, index=idx
            ))
            idx += 1
            continue

        # polygon(x1,y1,x2,y2,...,xn,yn)
        m = re.match(r'(?:-\s*)?polygon\(([^)]+)\)', line, re.IGNORECASE)
        if m:
            parts = [p.strip() for p in m.group(1).split(',')]
            regions.append(ParsedRegion(
                shape='polygon', coord_system=coord_system,
                raw_params=parts, label=label, color=color, index=idx
            ))
            idx += 1
            continue

        # box(x,y,w,h,theta)
        m = re.match(r'(?:-\s*)?box\(([^)]+)\)', line, re.IGNORECASE)
        if m:
            parts = [p.strip() for p in m.group(1).split(',')]
            regions.append(ParsedRegion(
                shape='box', coord_system=coord_system,
                raw_params=parts, label=label, color=color, index=idx
            ))
            idx += 1
            continue

    # Assign default labels to unlabeled regions
    for i, reg in enumerate(regions):
        if not reg.label:
            reg.label = f"Region {i + 1} ({reg.shape})"

    return regions


def _parse_angle(s: str) -> float:
    """Parse a DS9 angle/coordinate string to degrees or pixels."""
    s = s.strip()

    # Sexagesimal: HH:MM:SS.ss or DD:MM:SS.ss
    if ':' in s:
        parts = s.split(':')
        if len(parts) == 3:
            sign = -1 if s.startswith('-') else 1
            d = abs(float(parts[0]))
            m = float(parts[1])
            sec = float(parts[2])
            return sign * (d + m / 60.0 + sec / 3600.0)

    # With unit suffix
    if s.endswith('"'):
        return float(s[:-1]) / 3600.0
    if s.endswith("'"):
        return float(s[:-1]) / 60.0
    if s.endswith('d'):
        return float(s[:-1])

    return float(s)


def _get_pixel_scale(wcs_obj: WCS) -> float:
    """Get pixel scale in degrees/pixel from WCS."""
    if hasattr(wcs_obj.wcs, 'cdelt') and wcs_obj.wcs.cdelt[1] != 0:
        return np.abs(wcs_obj.wcs.cdelt[1])
    try:
        cd = wcs_obj.pixel_scale_matrix
        return np.sqrt(np.abs(np.linalg.det(cd)))
    except Exception:
        return 1.0 / 3600.0  # fallback: 1 arcsec/pix


# =============================================================================
# Region → Pixel Coordinates Conversion
# =============================================================================

def region_to_pixel_representation(region: ParsedRegion,
                                   wcs_obj: Optional[WCS] = None
                                   ) -> dict:
    """
    Convert a ParsedRegion to pixel-coordinate representation.

    Returns a dict with:
        'type': 'aperture' or 'mask'
        For 'aperture' (circle/ellipse):
            'shape', 'center', 'params'
        For 'mask' (polygon/box):
            'vertices_xy': np.ndarray of (N,2) pixel vertices
            'center': (cx, cy) centroid
    """
    if region.shape in ('circle', 'ellipse'):
        return _convert_aperture_region(region, wcs_obj)
    elif region.shape == 'polygon':
        return _convert_polygon_region(region, wcs_obj)
    elif region.shape == 'box':
        return _convert_box_region(region, wcs_obj)
    else:
        raise ValueError(f"Unsupported shape: {region.shape}")


def _convert_aperture_region(region: ParsedRegion,
                             wcs_obj: Optional[WCS]) -> dict:
    """Convert circle/ellipse to pixel aperture params."""
    p = region.raw_params
    cs = region.coord_system

    if cs == 'wcs':
        ra_deg = _parse_angle(p[0])
        if ':' in p[0]:
            ra_deg *= 15.0
        dec_deg = _parse_angle(p[1])

        if wcs_obj is None:
            raise ValueError("WCS required for WCS regions")

        cx, cy = wcs_obj.all_world2pix(ra_deg, dec_deg, 0)
        pix_scale = _get_pixel_scale(wcs_obj)

        if region.shape == 'circle':
            r_pix = _parse_angle(p[2]) / pix_scale
            return {'type': 'aperture', 'shape': 'circle',
                    'center': (float(cx), float(cy)), 'params': (r_pix,)}
        else:  # ellipse
            a_pix = _parse_angle(p[2]) / pix_scale
            b_pix = _parse_angle(p[3]) / pix_scale
            theta = np.radians(float(p[4])) if len(p) > 4 else 0.0
            return {'type': 'aperture', 'shape': 'ellipse',
                    'center': (float(cx), float(cy)),
                    'params': (a_pix, b_pix, theta)}
    else:
        # Pixel coords — DS9 is 1-indexed
        cx = float(p[0]) - 1
        cy = float(p[1]) - 1
        if region.shape == 'circle':
            r = float(p[2])
            return {'type': 'aperture', 'shape': 'circle',
                    'center': (cx, cy), 'params': (r,)}
        else:
            a, b = float(p[2]), float(p[3])
            theta = np.radians(float(p[4])) if len(p) > 4 else 0.0
            return {'type': 'aperture', 'shape': 'ellipse',
                    'center': (cx, cy), 'params': (a, b, theta)}


def _convert_polygon_region(region: ParsedRegion,
                            wcs_obj: Optional[WCS]) -> dict:
    """Convert polygon vertices to pixel coordinates."""
    p = region.raw_params
    cs = region.coord_system

    # Polygon params: x1,y1,x2,y2,...,xn,yn
    if len(p) < 6:
        raise ValueError(f"Polygon needs >= 3 vertices, got {len(p)//2}")

    n_verts = len(p) // 2
    vertices = np.zeros((n_verts, 2))

    if cs == 'wcs':
        if wcs_obj is None:
            raise ValueError("WCS required for WCS polygon")

        ras = []
        decs = []
        for i in range(n_verts):
            ra_deg = _parse_angle(p[2*i])
            if ':' in p[2*i]:
                ra_deg *= 15.0
            dec_deg = _parse_angle(p[2*i + 1])
            ras.append(ra_deg)
            decs.append(dec_deg)

        pix_x, pix_y = wcs_obj.all_world2pix(ras, decs, 0)
        vertices[:, 0] = pix_x
        vertices[:, 1] = pix_y
    else:
        for i in range(n_verts):
            vertices[i, 0] = float(p[2*i]) - 1     # DS9 1-indexed
            vertices[i, 1] = float(p[2*i + 1]) - 1

    centroid = vertices.mean(axis=0)
    return {'type': 'mask', 'shape': 'polygon',
            'vertices_xy': vertices,
            'center': (float(centroid[0]), float(centroid[1]))}


def _convert_box_region(region: ParsedRegion,
                        wcs_obj: Optional[WCS]) -> dict:
    """Convert box region to polygon vertices in pixel coords."""
    p = region.raw_params
    cs = region.coord_system

    if cs == 'wcs':
        if wcs_obj is None:
            raise ValueError("WCS required for WCS box")

        ra_deg = _parse_angle(p[0])
        if ':' in p[0]:
            ra_deg *= 15.0
        dec_deg = _parse_angle(p[1])

        cx, cy = wcs_obj.all_world2pix(ra_deg, dec_deg, 0)
        pix_scale = _get_pixel_scale(wcs_obj)

        w_pix = _parse_angle(p[2]) / pix_scale
        h_pix = _parse_angle(p[3]) / pix_scale
        theta = np.radians(float(p[4])) if len(p) > 4 else 0.0
    else:
        cx = float(p[0]) - 1
        cy = float(p[1]) - 1
        w_pix = float(p[2])
        h_pix = float(p[3])
        theta = np.radians(float(p[4])) if len(p) > 4 else 0.0

    # Generate 4 corner vertices of the rotated box
    hw, hh = w_pix / 2.0, h_pix / 2.0
    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t],
                     [sin_t,  cos_t]])

    vertices = (rot @ corners_local.T).T + np.array([float(cx), float(cy)])

    return {'type': 'mask', 'shape': 'box',
            'vertices_xy': vertices,
            'center': (float(cx), float(cy))}


# =============================================================================
# Polygon/Mask Geometry Utilities
# =============================================================================

def simplify_polygon(vertices: np.ndarray, tolerance: float = 0.5
                     ) -> np.ndarray:
    """
    Reduce polygon vertex count using the Ramer-Douglas-Peucker algorithm.

    DS9 contour-derived polygons can have hundreds or thousands of vertices
    tracing smooth curves at sub-pixel resolution. Most of these are
    redundant — a 2000-vertex contour reduces to ~80-100 vertices with
    <0.5 pixel deviation, making subsequent masking fast.

    Parameters:
        vertices: (N, 2) array of polygon vertices
        tolerance: maximum perpendicular distance (pixels) a vertex can
                   deviate from the simplified line. Default 0.5 pixels.

    Returns:
        Simplified vertices (M, 2) where M <= N
    """
    if len(vertices) <= 4:
        return vertices

    # Iterative RDP (avoids recursion depth issues with huge contours)
    keep = np.zeros(len(vertices), dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, len(vertices) - 1)]

    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue

        p1 = vertices[start]
        p2 = vertices[end]
        segment = p2 - p1
        seg_len_sq = np.dot(segment, segment)

        if seg_len_sq == 0:
            dists = np.linalg.norm(vertices[start+1:end] - p1, axis=1)
        else:
            t = np.dot(vertices[start+1:end] - p1, segment) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            projections = p1 + np.outer(t, segment)
            dists = np.linalg.norm(vertices[start+1:end] - projections, axis=1)

        max_idx = np.argmax(dists) + start + 1
        max_dist = dists[max_idx - start - 1]

        if max_dist > tolerance:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))

    return vertices[keep]


def make_polygon_mask(vertices_xy: np.ndarray, shape: Tuple[int, int]
                      ) -> np.ndarray:
    """
    Create a boolean mask from polygon vertices.

    Simplifies high-vertex-count polygons (contours) via RDP, then
    rasterizes using MplPath.contains_points within the bounding box.

    Parameters:
        vertices_xy: (N, 2) array of (x, y) pixel coordinates
        shape: (ny, nx) image dimensions

    Returns:
        Boolean mask array of shape (ny, nx)
    """
    ny, nx = shape

    # Simplify contours — the key speedup for complex polygons
    verts = simplify_polygon(vertices_xy, tolerance=0.5)

    # Bounding box crop — only test pixels near the polygon
    path = MplPath(verts)
    x_min = max(0, int(np.floor(verts[:, 0].min())) - 1)
    x_max = min(nx, int(np.ceil(verts[:, 0].max())) + 2)
    y_min = max(0, int(np.floor(verts[:, 1].min())) - 1)
    y_max = min(ny, int(np.ceil(verts[:, 1].max())) + 2)

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    sub_mask = path.contains_points(points).reshape(y_max - y_min, x_max - x_min)

    mask = np.zeros((ny, nx), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = sub_mask
    return mask


def dilate_polygon(vertices_xy: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale polygon vertices outward from centroid by a multiplicative factor.

    This creates the inner/outer boundaries for a polygon "annulus"
    used in local background estimation.

    Parameters:
        vertices_xy: (N, 2) array of vertex positions
        factor: scale factor (>1 expands, <1 shrinks)

    Returns:
        Scaled vertices (N, 2)
    """
    centroid = vertices_xy.mean(axis=0)
    delta = vertices_xy - centroid
    return centroid + delta * factor


def make_polygon_annulus_mask(vertices_xy: np.ndarray,
                              inner_factor: float, outer_factor: float,
                              shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a boolean annulus mask by dilating the polygon.

    The annulus is the region between the inner and outer dilated polygons.
    The source polygon itself is excluded.
    """
    inner_verts = dilate_polygon(vertices_xy, inner_factor)
    outer_verts = dilate_polygon(vertices_xy, outer_factor)

    outer_mask = make_polygon_mask(outer_verts, shape)
    inner_mask = make_polygon_mask(inner_verts, shape)

    # Annulus = inside outer but outside inner
    return outer_mask & ~inner_mask


# =============================================================================
# Photometry — unified interface for all region types
# =============================================================================

def do_photometry(data: np.ndarray, pixel_rep: dict,
                  bg_annulus_factor: Optional[Tuple[float, float]] = (1.5, 2.5)
                  ) -> Tuple[float, float, int]:
    """
    Perform aperture/mask photometry on data for any region type.

    Parameters:
        data: 2D image array
        pixel_rep: output of region_to_pixel_representation()
        bg_annulus_factor: (inner, outer) scale factors for background

    Returns:
        (net_flux, flux_error, n_source_pixels)
    """
    if pixel_rep['type'] == 'aperture':
        return _photometry_aperture(data, pixel_rep, bg_annulus_factor)
    elif pixel_rep['type'] == 'mask':
        return _photometry_mask(data, pixel_rep, bg_annulus_factor)
    else:
        raise ValueError(f"Unknown pixel_rep type: {pixel_rep['type']}")


def _photometry_aperture(data: np.ndarray, pixel_rep: dict,
                         bg_annulus_factor) -> Tuple[float, float, int]:
    """Photometry for circle/ellipse using photutils or fallback."""
    shape = pixel_rep['shape']
    center = pixel_rep['center']
    params = pixel_rep['params']

    if HAS_PHOTUTILS:
        return _photometry_aperture_photutils(data, shape, center, params,
                                              bg_annulus_factor)
    else:
        return _photometry_aperture_simple(data, shape, center, params,
                                           bg_annulus_factor)


def _photometry_aperture_photutils(data, shape, center, params,
                                   bg_annulus_factor):
    """Circle/ellipse photometry via photutils. Returns (flux, error, npix)."""
    cx, cy = center

    if shape == 'circle':
        r = params[0]
        aperture = CircularAperture((cx, cy), r=r)
        if bg_annulus_factor:
            from photutils.aperture import CircularAnnulus
            bg_aper = CircularAnnulus((cx, cy),
                                     r_in=r * bg_annulus_factor[0],
                                     r_out=r * bg_annulus_factor[1])
    elif shape == 'ellipse':
        a, b, theta = params
        aperture = EllipticalAperture((cx, cy), a=a, b=b, theta=theta)
        if bg_annulus_factor:
            from photutils.aperture import EllipticalAnnulus
            bg_aper = EllipticalAnnulus((cx, cy),
                                       a_in=a * bg_annulus_factor[0],
                                       a_out=a * bg_annulus_factor[1],
                                       b_in=b * bg_annulus_factor[0],
                                       b_out=b * bg_annulus_factor[1],
                                       theta=theta)
    else:
        raise ValueError(f"Unsupported aperture shape: {shape}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src_stats = ApertureStats(data, aperture)
        src_sum = src_stats.sum
        npix = (src_stats.sum_aper_area.value
                if hasattr(src_stats.sum_aper_area, 'value')
                else src_stats.sum_aper_area)
        npix = int(round(npix))

        if bg_annulus_factor:
            bg_stats = ApertureStats(data, bg_aper)
            bg_median = bg_stats.median
            if np.isfinite(bg_median):
                net_flux = src_sum - bg_median * npix
                bg_std = bg_stats.std
                if np.isfinite(bg_std):
                    err = np.sqrt(np.abs(net_flux) + (bg_std ** 2) * npix)
                else:
                    err = np.sqrt(np.abs(net_flux))
                return float(net_flux), float(err), npix

        err = np.sqrt(np.abs(src_sum)) if np.isfinite(src_sum) else 0.0
        return float(src_sum), float(err), npix


def _photometry_aperture_simple(data, shape, center, params,
                                bg_annulus_factor):
    """Circle/ellipse photometry via boolean masks (no photutils)."""
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]
    cx, cy = center

    if shape == 'circle':
        r = params[0]
        dist2 = (xx - cx)**2 + (yy - cy)**2
        src_mask = dist2 <= r**2
        if bg_annulus_factor:
            bg_mask = ((dist2 >= (r * bg_annulus_factor[0])**2) &
                       (dist2 <= (r * bg_annulus_factor[1])**2))
        else:
            bg_mask = None
    elif shape == 'ellipse':
        a, b, theta = params
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        dx, dy = xx - cx, yy - cy
        xr = dx * cos_t - dy * sin_t
        yr = dx * sin_t + dy * cos_t
        src_mask = (xr / a)**2 + (yr / b)**2 <= 1.0
        if bg_annulus_factor:
            f_in, f_out = bg_annulus_factor
            inner = (xr / (a * f_in))**2 + (yr / (b * f_in))**2 >= 1.0
            outer = (xr / (a * f_out))**2 + (yr / (b * f_out))**2 <= 1.0
            bg_mask = inner & outer
        else:
            bg_mask = None
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return _sum_with_background(data, src_mask, bg_mask)


def _photometry_mask(data: np.ndarray, pixel_rep: dict,
                     bg_annulus_factor) -> Tuple[float, float, int]:
    """
    Photometry for polygon/box regions using boolean masks.

    Pipeline:
      1. Simplify vertices (RDP, 0.5 px tolerance)
      2. Compute bounding box of outermost annulus
      3. Build pixel grid once, test all masks against it
    """
    vertices = pixel_rep['vertices_xy']
    ny, nx = data.shape

    # Simplify contours — reduces ~2000 vertices to ~80-100
    verts = simplify_polygon(vertices, tolerance=0.5)

    # Determine the outermost boundary for the bounding box
    if bg_annulus_factor:
        outer_verts = dilate_polygon(verts, bg_annulus_factor[1])
        all_verts = np.vstack([verts, outer_verts])
    else:
        all_verts = verts

    # Bounding box clipped to image
    x_min = max(0, int(np.floor(all_verts[:, 0].min())) - 1)
    x_max = min(nx, int(np.ceil(all_verts[:, 0].max())) + 2)
    y_min = max(0, int(np.floor(all_verts[:, 1].min())) - 1)
    y_max = min(ny, int(np.ceil(all_verts[:, 1].max())) + 2)

    sub_h = y_max - y_min
    sub_w = x_max - x_min
    if sub_h <= 0 or sub_w <= 0:
        return 0.0, 0.0

    # Shift vertices into subimage coordinates
    offset = np.array([x_min, y_min])
    sub_verts = verts - offset

    # Build the pixel grid once — reused for all masks
    yy, xx = np.mgrid[:sub_h, :sub_w]
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Source mask
    src_mask = MplPath(sub_verts).contains_points(points).reshape(sub_h, sub_w)

    # Background annulus mask
    bg_mask = None
    if bg_annulus_factor:
        inner_sub = dilate_polygon(sub_verts, bg_annulus_factor[0])
        outer_sub = dilate_polygon(sub_verts, bg_annulus_factor[1])
        inner_mask = MplPath(inner_sub).contains_points(points).reshape(sub_h, sub_w)
        outer_mask = MplPath(outer_sub).contains_points(points).reshape(sub_h, sub_w)
        bg_mask = outer_mask & ~inner_mask

    # Extract subimage data
    sub_data = data[y_min:y_max, x_min:x_max]

    return _sum_with_background(sub_data, src_mask, bg_mask)


def _sum_with_background(data: np.ndarray, src_mask: np.ndarray,
                         bg_mask: Optional[np.ndarray]
                         ) -> Tuple[float, float, int]:
    """
    Sum source flux with optional local background subtraction.

    Background level is estimated as the sigma-clipped median
    in the annulus to be robust against contaminating sources.

    Returns:
        (net_flux, flux_error, n_source_pixels)
    """
    src_data = data[src_mask]
    src_data = src_data[np.isfinite(src_data)]
    if len(src_data) == 0:
        return 0.0, 0.0, 0

    src_sum = np.sum(src_data)
    npix = len(src_data)

    if bg_mask is not None:
        bg_data = data[bg_mask]
        bg_data = bg_data[np.isfinite(bg_data)]

        if len(bg_data) > 5:
            # Simple sigma clip: 2 iterations, 3-sigma
            bg_med = np.median(bg_data)
            bg_std = np.std(bg_data)
            for _ in range(2):
                clip = np.abs(bg_data - bg_med) < 3.0 * bg_std
                if clip.sum() > 3:
                    bg_data = bg_data[clip]
                    bg_med = np.median(bg_data)
                    bg_std = np.std(bg_data)

            net_flux = src_sum - bg_med * npix
            err = np.sqrt(np.abs(net_flux) + (bg_std ** 2) * npix)
            return float(net_flux), float(err), npix

    err = np.sqrt(np.abs(src_sum))
    return float(src_sum), float(err), npix


# =============================================================================
# Frame Discovery & Data Caching
# =============================================================================

def load_frame_info(ds9: DS9XPA, frame_num: int,
                    custom_wavelengths: Optional[Dict] = None
                    ) -> Optional[FrameInfo]:
    """Load and cache data for one frame."""
    original_frame = ds9.get_current_frame()
    ds9.set_frame(frame_num)

    try:
        filename = ds9.get_filename()
    except RuntimeError:
        ds9.set_frame(original_frame)
        return None

    if not filename or not os.path.isfile(filename):
        print(f"  Frame {frame_num}: file not found ({filename})")
        ds9.set_frame(original_frame)
        return None

    try:
        with fits.open(filename) as hdul:
            # Prefer SCI extension, then first image HDU
            hdu = None
            for h in hdul:
                if h.name == 'SCI' and h.data is not None:
                    hdu = h
                    break
            if hdu is None:
                for h in hdul:
                    if h.data is not None and h.data.ndim >= 2:
                        hdu = h
                        break

            if hdu is None:
                print(f"  Frame {frame_num}: no image data in {filename}")
                ds9.set_frame(original_frame)
                return None

            data = hdu.data.astype(float)
            header = hdu.header
            if hdu is not hdul[0]:
                for key in FILTER_HEADER_KEYS + BUNIT_HEADER_KEYS + ['PIXAR_SR']:
                    if key not in header and key in hdul[0].header:
                        header[key] = hdul[0].header[key]

            wcs_obj = WCS(header, naxis=2)

    except Exception as e:
        print(f"  Frame {frame_num}: error reading {filename}: {e}")
        ds9.set_frame(original_frame)
        return None

    # Determine filter
    filter_name = 'UNKNOWN'
    for key in FILTER_HEADER_KEYS:
        if key in header:
            val = str(header[key]).strip().upper()
            if val and val != 'CLEAR' and val != 'NONE':
                filter_name = val
                break

    # Determine wavelength
    wave_db = {**FILTER_WAVELENGTHS}
    if custom_wavelengths:
        wave_db.update(custom_wavelengths)

    wavelength = wave_db.get(filter_name, 0.0)
    if wavelength == 0.0:
        for k, v in wave_db.items():
            if k in filter_name or filter_name in k:
                wavelength = v
                break

    if wavelength == 0.0:
        for wkey in ['WAVELEN', 'WAVELENG', 'CENTRWV', 'PHOTPLAM']:
            if wkey in header:
                w = float(header[wkey])
                if w > 1000:
                    wavelength = w / 1e4
                elif w > 10:
                    wavelength = w
                else:
                    wavelength = w
                break

    bunit = ''
    for key in BUNIT_HEADER_KEYS:
        if key in header:
            bunit = str(header[key]).strip()
            break

    pixar_sr = header.get('PIXAR_SR', 0.0)

    info = FrameInfo(
        frame_num=frame_num,
        filename=filename,
        filter_name=filter_name,
        wavelength_um=wavelength,
        data=data,
        header=header,
        wcs=wcs_obj,
        bunit=bunit,
        pixar_sr=pixar_sr,
        label=f"{filter_name}" + (f" ({wavelength:.1f} \u00b5m)"
                                   if wavelength > 0 else ""),
    )

    ds9.set_frame(original_frame)
    return info


# =============================================================================
# SED Extraction — single and multi-region
# =============================================================================

@dataclass
class SEDResult:
    """SED measurement for one region across all filters."""
    label: str
    color: str
    wavelengths: np.ndarray
    fluxes: np.ndarray
    errors: np.ndarray
    filter_labels: List[str]


class SEDExtractor:
    """Extract SEDs from one or more DS9 regions across multiple frames."""

    def __init__(self, frames: List[FrameInfo], coord_system: str = 'wcs',
                 bg_annulus_factor: Optional[Tuple[float, float]] = (1.5, 2.5),
                 flux_unit: str = 'native'):
        self.frames = sorted(frames, key=lambda f: f.wavelength_um)
        self.coord_system = coord_system
        self.bg_annulus_factor = bg_annulus_factor
        self.flux_unit = flux_unit

    def extract_all(self, region_text: str) -> List[SEDResult]:
        """
        Extract SEDs for ALL regions found in the region text.

        Returns a list of SEDResult, one per region.
        """
        regions = parse_all_ds9_regions(region_text, self.coord_system)
        if not regions:
            return []

        results = []
        for region in regions:
            sed = self._extract_one(region)
            if sed is not None and len(sed.wavelengths) > 0:
                results.append(sed)

        return results

    def _extract_one(self, region: ParsedRegion) -> Optional[SEDResult]:
        """Extract SED for a single region."""
        wavelengths = []
        fluxes = []
        errors = []
        filter_labels = []

        for frame in self.frames:
            if frame.wavelength_um <= 0:
                continue

            # Convert region to pixel representation for this frame
            try:
                pixel_rep = region_to_pixel_representation(
                    region, wcs_obj=frame.wcs
                )
            except Exception as e:
                print(f"  Region '{region.label}' conversion failed for "
                      f"frame {frame.frame_num}: {e}")
                continue

            # Bounds check
            ny, nx = frame.data.shape
            if pixel_rep['type'] == 'aperture':
                cx, cy = pixel_rep['center']
                if cx < 0 or cx >= nx or cy < 0 or cy >= ny:
                    continue
            elif pixel_rep['type'] == 'mask':
                verts = pixel_rep['vertices_xy']
                # Check that at least the centroid is in-frame
                ctr = verts.mean(axis=0)
                if ctr[0] < -nx * 0.5 or ctr[0] >= nx * 1.5:
                    continue
                if ctr[1] < -ny * 0.5 or ctr[1] >= ny * 1.5:
                    continue

            # Photometry
            try:
                flux, err, npix = do_photometry(
                    frame.data, pixel_rep, self.bg_annulus_factor
                )
            except Exception as e:
                print(f"  Photometry failed for '{region.label}' in "
                      f"frame {frame.frame_num}: {e}")
                continue

            if npix == 0:
                continue

            # Unit conversion
            if self.flux_unit == 'MJy/sr':
                # Average surface brightness: sum / npix
                # Native JWST pixels are MJy/sr, so the mean pixel value
                # after background subtraction is the mean surface brightness
                flux = flux / npix
                err = err / npix
            elif self.flux_unit == 'Jy' and frame.pixar_sr > 0:
                flux = flux * frame.pixar_sr * 1e6
                err = err * frame.pixar_sr * 1e6
            elif self.flux_unit == 'mJy' and frame.pixar_sr > 0:
                flux = flux * frame.pixar_sr * 1e9
                err = err * frame.pixar_sr * 1e9

            wavelengths.append(frame.wavelength_um)
            fluxes.append(flux)
            errors.append(err)
            filter_labels.append(frame.label)

        if not wavelengths:
            return None

        return SEDResult(
            label=region.label,
            color=region.color,
            wavelengths=np.array(wavelengths),
            fluxes=np.array(fluxes),
            errors=np.array(errors),
            filter_labels=filter_labels,
        )


# =============================================================================
# Real-Time Multi-Region SED Plot
# =============================================================================

# Color cycle for multiple regions
REGION_COLORS = [
    '#2176AE', '#E63946', '#2A9D8F', '#E9C46A', '#F4A261',
    '#264653', '#D62828', '#6A4C93', '#1B998B', '#FF6B6B',
]

# Map DS9 color names to pleasant matplotlib equivalents
DS9_COLOR_MAP = {
    'green': '#2A9D8F', 'red': '#E63946', 'blue': '#2176AE',
    'cyan': '#17BECF', 'magenta': '#D62828', 'yellow': '#E9C46A',
    'white': '#555555', 'black': '#333333',
}


class SEDPlotter:
    """Matplotlib-based SED plot with threaded XPA polling.

    Architecture:
        Worker thread: polls DS9 via XPA, runs photometry, pushes results
        Main thread:   fast timer checks for new results and redraws

    This keeps the matplotlib window responsive while dragging regions
    in DS9, since XPA calls and photometry never block the GUI.
    """

    def __init__(self, ds9: DS9XPA, extractor: SEDExtractor,
                 coord_system: str = 'wcs', log_scale: bool = False,
                 poll_interval: int = 100, save_csv: Optional[str] = None):
        self.ds9 = ds9
        self.extractor = extractor
        self.coord_system = coord_system
        self.poll_interval = poll_interval
        self.save_csv = save_csv

        # Thread-safe communication
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()

        # Track plot objects for cleanup
        self._plot_objects = []

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.fig.canvas.manager.set_window_title('DS9 Interactive SED')

        self.ax.set_xlabel('Wavelength (\u00b5m)', fontsize=12)
        if extractor.flux_unit == 'Jy':
            ylabel = 'Flux Density (Jy)'
        elif extractor.flux_unit == 'mJy':
            ylabel = 'Flux Density (mJy)'
        elif extractor.flux_unit == 'MJy/sr':
            ylabel = 'Surface Brightness (MJy/sr)'
        else:
            ylabel = 'Aperture Sum (native units)'
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title('SED \u2014 draw regions in DS9...', fontsize=13)

        if log_scale:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
            self.ax.xaxis.set_major_formatter(ScalarFormatter())
            self.ax.xaxis.set_minor_formatter(ScalarFormatter())

        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.18, right=0.78)

        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=8, color='gray', family='monospace'
        )

    def _worker_loop(self):
        """Background thread: poll DS9, debounce, then compute SEDs.

        Debouncing is critical for contour polygons. While dragging, DS9
        fires new region coordinates on every mouse event (dozens/sec).
        Computing a full SED for each intermediate position is wasteful
        and causes the GIL-bound computation to starve the GUI thread.

        Instead: poll fast, but only compute after the region has been
        stable for `debounce_ms` milliseconds. While dragging, nothing
        computes and the GUI stays responsive.
        """
        last_region_text = None
        last_change_time = 0.0
        pending_region_text = None
        debounce_sec = 0.25  # 250ms — feels instant after releasing mouse
        poll_sec = 0.05      # 50ms — responsive to changes

        while not self._stop_event.is_set():
            # Poll DS9 for current regions
            try:
                region_text = self.ds9.get_regions(
                    coord_system=self.coord_system
                )
            except Exception:
                self._stop_event.wait(poll_sec)
                continue

            now = time.time()

            if not region_text:
                self._stop_event.wait(poll_sec)
                continue

            # Detect change
            if region_text != pending_region_text:
                # Region moved — reset the debounce timer
                pending_region_text = region_text
                last_change_time = now
                self._stop_event.wait(poll_sec)
                continue

            # Region unchanged — check if debounce period has elapsed
            if now - last_change_time < debounce_sec:
                self._stop_event.wait(poll_sec)
                continue

            # Debounce passed and this is a new region to compute
            if pending_region_text == last_region_text:
                self._stop_event.wait(poll_sec)
                continue

            # Compute SED
            last_region_text = pending_region_text
            t0 = time.time()
            sed_results = self.extractor.extract_all(pending_region_text)
            dt = time.time() - t0

            # Push to GUI thread — drop stale results
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break
            self._result_queue.put((sed_results, dt))

            self._stop_event.wait(poll_sec)

    def _gui_check(self, frame_number):
        """FuncAnimation callback — check queue and redraw if new results."""
        try:
            sed_results, dt = self._result_queue.get_nowait()
        except queue.Empty:
            return

        if not sed_results:
            self.ax.set_title(
                'SED \u2014 no valid data (check regions & filters)', fontsize=13
            )
            self.fig.canvas.draw_idle()
            return

        self._redraw(sed_results)

        n_regions = len(sed_results)
        n_filters = max(len(s.wavelengths) for s in sed_results)

        self.status_text.set_text(
            f'{n_regions} region{"s" if n_regions > 1 else ""} | '
            f'{n_filters} filters | {dt:.2f}s | {time.strftime("%H:%M:%S")}'
        )
        self.fig.canvas.draw_idle()

        if self.save_csv:
            self._save_csv(sed_results)

    def _redraw(self, sed_results: List[SEDResult]):
        """Redraw all SED curves."""
        ax = self.ax

        # Remove old plot objects
        for line, errbar, anns in self._plot_objects:
            line.remove()
            if errbar is not None:
                errbar.remove()
            for ann in anns:
                ann.remove()
        self._plot_objects = []

        # Remove old legend
        legend = ax.get_legend()
        if legend:
            legend.remove()

        # Plot each region's SED
        all_waves = []
        all_fluxes = []

        for i, sed in enumerate(sed_results):
            # Color: prefer DS9 region color, fall back to cycle
            color = sed.color if sed.color else ''
            if color.lower() in DS9_COLOR_MAP:
                color = DS9_COLOR_MAP[color.lower()]
            elif not color or color.startswith('#') is False:
                color = REGION_COLORS[i % len(REGION_COLORS)]

            line, = ax.plot(
                sed.wavelengths, sed.fluxes,
                'o-', color=color, markersize=7, linewidth=1.5,
                label=sed.label, zorder=3 + i
            )

            errbar = ax.errorbar(
                sed.wavelengths, sed.fluxes, yerr=sed.errors,
                fmt='none', ecolor=color, elinewidth=1.5, capsize=3,
                alpha=0.7, zorder=2 + i
            )

            # Filter labels — only for single region to avoid clutter
            anns = []
            if len(sed_results) == 1:
                for w, f, lbl in zip(sed.wavelengths, sed.fluxes,
                                     sed.filter_labels):
                    ann = ax.annotate(
                        lbl.split('(')[0].strip(),
                        (w, f), textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=7, color='#555555',
                        fontweight='bold'
                    )
                    anns.append(ann)

            self._plot_objects.append((line, errbar, anns))

            all_waves.extend(sed.wavelengths)
            all_fluxes.extend(sed.fluxes)

        # Legend for multi-region
        if len(sed_results) > 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.01, 1.0),
                fontsize=9, framealpha=0.9, borderaxespad=0
            )

        # Rescale axes
        all_waves = np.array(all_waves)
        all_fluxes = np.array(all_fluxes)
        all_fluxes_valid = all_fluxes[np.isfinite(all_fluxes)]

        if len(all_waves) > 0:
            wmin, wmax = all_waves.min(), all_waves.max()
            if ax.get_xscale() == 'log':
                ax.set_xlim(wmin * 0.8, wmax * 1.3)
            else:
                dw = max(wmax - wmin, 1.0) * 0.15
                ax.set_xlim(wmin - dw, wmax + dw)

        if len(all_fluxes_valid) > 0:
            fmin, fmax = all_fluxes_valid.min(), all_fluxes_valid.max()
            if ax.get_yscale() == 'log' and fmin > 0:
                ax.set_ylim(fmin * 0.3, fmax * 3.0)
            else:
                df = max(fmax - fmin, abs(fmax) * 0.1) * 0.25
                ax.set_ylim(fmin - df, fmax + df)

        title = 'Spectral Energy Distribution'
        if len(sed_results) > 1:
            title += f'  ({len(sed_results)} regions)'
        ax.set_title(title, fontsize=13)

    def _save_csv(self, sed_results: List[SEDResult]):
        """Save all SEDs to CSV."""
        try:
            with open(self.save_csv, 'w') as f:
                f.write('region,wavelength_um,flux,flux_err,filter\n')
                for sed in sed_results:
                    for w, fl, e, lbl in zip(sed.wavelengths, sed.fluxes,
                                             sed.errors, sed.filter_labels):
                        safe_label = sed.label.replace(',', ';')
                        f.write(f'{safe_label},{w:.4f},{fl:.6e},{e:.6e},{lbl}\n')
        except Exception as e:
            print(f"  Warning: could not save CSV: {e}")

    def run(self):
        """Start the worker thread and interactive plot."""
        import signal

        # Handle Ctrl+C gracefully (plt.show blocks signal handling
        # on some backends)
        def _signal_handler(signum, frame):
            print("\n  Shutting down...")
            self._stop_event.set()
            plt.close(self.fig)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # Start background XPA polling + photometry thread
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

        # GUI-side: fast timer just checks the queue (no computation)
        self.anim = FuncAnimation(
            self.fig, self._gui_check,
            interval=50,  # 50ms — just checking a queue, very cheap
            cache_frame_data=False,
            save_count=0,
        )

        try:
            plt.show()
        finally:
            self._stop_event.set()
            worker.join(timeout=2)


# =============================================================================
# DS9 Analysis File Generator
# =============================================================================

def generate_analysis_file(script_path: str, output_path: str,
                           coord_system: str = 'wcs'):
    """Generate a .ds9 analysis descriptor file.

    The launch command fully detaches from DS9's process group and
    redirects all I/O so DS9 doesn't block waiting on pipes.
    Uses setsid on Linux, nohup on macOS.
    """
    import platform
    if platform.system() == 'Darwin':
        # macOS: setsid doesn't exist; nohup + & detaches adequately
        detach = 'nohup'
    else:
        # Linux: setsid creates a new session, fully detaching
        detach = 'setsid'

    def _cmd(extra_flags=''):
        flags = f'--coord-system {coord_system}'
        if extra_flags:
            flags += f' {extra_flags}'
        return (f'{detach} python3 {script_path} {flags} '
                f'</dev/null >/tmp/ds9_sed_tool.log 2>&1 &')

    content = f"""# DS9 Interactive SED Tool
# Install: Edit > Preferences > Analysis > Analysis File > Load
# Or: ds9 -analysis load {output_path}
# Log output goes to /tmp/ds9_sed_tool.log

Interactive SED Plotter (WCS)
*
menu
{_cmd()}

Interactive SED Plotter (Jy, log)
*
menu
{_cmd('--flux-unit Jy --log')}
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"DS9 analysis file written to: {output_path}")
    print(f"  Platform: {platform.system()} (using {detach})")
    print(f"  Log output: /tmp/ds9_sed_tool.log")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DS9 Interactive SED Tool \u2014 real-time multi-region SEDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic — all regions, native units
  python ds9_sed_tool.py

  # JWST MIRI with Jy conversion, log axes
  python ds9_sed_tool.py --flux-unit Jy --log

  # Single-region mode (first region only)
  python ds9_sed_tool.py --single

  # No background subtraction, pixel coordinates
  python ds9_sed_tool.py --coord-system image --no-bg

  # Custom wavelengths and CSV output
  python ds9_sed_tool.py --wavelength F1=7.7 F2=10.0 --save-csv sed.csv

  # Generate DS9 analysis menu entry
  python ds9_sed_tool.py --generate-ds9-file sed_tool.ds9

Region labeling in DS9:
  Double-click a region > Properties > Text
  Type a name (e.g., "ejecta knot", "CSM filament")
  Each labeled region gets its own SED curve with matching color.
        """
    )
    parser.add_argument('--target', default='ds9',
                        help='XPA target name (default: ds9)')
    parser.add_argument('--coord-system', choices=['wcs', 'image'],
                        default='wcs',
                        help='Region coordinate system (default: wcs)')
    parser.add_argument('--bg-annulus', nargs=2, type=float, default=[1.5, 2.5],
                        metavar=('INNER', 'OUTER'),
                        help='Background annulus inner/outer scale factors '
                             '(default: 1.5 2.5). For polygons, vertices are '
                             'dilated from centroid by these factors.')
    parser.add_argument('--no-bg', action='store_true',
                        help='Disable background subtraction')
    parser.add_argument('--flux-unit', choices=['native', 'Jy', 'mJy', 'MJy/sr'],
                        default='native',
                        help='Flux unit for plot. native=aperture sum, '
                             'Jy/mJy=integrated flux density, '
                             'MJy/sr=mean surface brightness (default: native)')
    parser.add_argument('--log', action='store_true',
                        help='Use log-log axes')
    parser.add_argument('--poll-interval', type=int, default=100,
                        help='XPA poll interval in ms (default: 100)')
    parser.add_argument('--save-csv', default=None,
                        help='Save SED data to CSV on each update')
    parser.add_argument('--single', action='store_true',
                        help='Single-region mode: use only the first region')
    parser.add_argument('--wavelength', nargs='*', default=None,
                        help='Custom filter=wavelength(um) mappings')
    parser.add_argument('--generate-ds9-file', default=None, metavar='PATH',
                        help='Generate a .ds9 analysis file and exit')
    parser.add_argument('--backend', default=None,
                        help='Matplotlib backend (default: auto-detect, '
                             'prefers Qt over Tk). Use QtAgg, Qt5Agg, '
                             'GTK3Agg, or TkAgg.')

    args = parser.parse_args()

    # Generate analysis file mode
    if args.generate_ds9_file:
        script_path = os.path.abspath(__file__)
        generate_analysis_file(script_path, args.generate_ds9_file,
                               args.coord_system)
        return

    # Parse custom wavelengths
    custom_wavelengths = {}
    if args.wavelength:
        for item in args.wavelength:
            if '=' in item:
                k, v = item.split('=', 1)
                custom_wavelengths[k.strip().upper()] = float(v.strip())

    # Connect to DS9
    print("DS9 Interactive SED Tool")
    print("=" * 50)
    print(f"  Matplotlib backend: {_selected_backend}")
    if _selected_backend == 'TkAgg':
        print("  ⚠ TkAgg backend: works from terminal, but will conflict")
        print("    with DS9 if launched from its Analysis menu.")
        print("    Install PyQt5/6 for best results: pip install PyQt5")
    print(f"  Connecting to DS9 (target='{args.target}')...")
    ds9 = DS9XPA(target=args.target)

    # Discover frames
    print("  Discovering frames...")
    frame_nums = ds9.discover_frames()
    print(f"  Found {len(frame_nums)} frames: {frame_nums}")

    # Load frame data
    frames = []
    for fn in frame_nums:
        print(f"  Loading frame {fn}...", end=' ')
        info = load_frame_info(ds9, fn, custom_wavelengths)
        if info:
            frames.append(info)
            status = f"{info.filter_name}"
            if info.wavelength_um > 0:
                status += f" @ {info.wavelength_um:.2f} \u00b5m"
            else:
                status += " (no wavelength \u2014 will skip)"
            if info.bunit:
                status += f" [{info.bunit}]"
            print(status)
        else:
            print("skipped (no data)")

    if not frames:
        print("\n  ERROR: No usable frames found.")
        sys.exit(1)

    valid = [f for f in frames if f.wavelength_um > 0]
    if not valid:
        print("\n  ERROR: No frames have recognized filter wavelengths.")
        print("  Use --wavelength FILTER=WAVELENGTH to set manually.")
        print(f"  Detected filters: {[f.filter_name for f in frames]}")
        sys.exit(1)

    print(f"\n  {len(valid)} frames with known wavelengths ready")
    print(f"  Wavelength range: {min(f.wavelength_um for f in valid):.1f}"
          f" \u2013 {max(f.wavelength_um for f in valid):.1f} \u00b5m")

    bg_factor = None if args.no_bg else tuple(args.bg_annulus)
    if bg_factor:
        print(f"  Background annulus: {bg_factor[0]:.1f}\u00d7 \u2013 "
              f"{bg_factor[1]:.1f}\u00d7 (dilated polygon for polygon/box)")
    else:
        print("  Background subtraction: OFF")

    mode = 'single' if args.single else 'multi'
    print(f"  Region mode: {mode}")
    print(f"  Supported shapes: circle, ellipse, polygon, box")
    raster = 'MplPath (bbox) + RDP simplification'
    print(f"  Polygon rasterizer: {raster}")
    print(f"  Coordinate system: {args.coord_system}")
    print(f"  Flux unit: {args.flux_unit}")
    print(f"  Poll interval: {args.poll_interval} ms")
    print()
    print("  Draw regions in DS9 \u2014 tag them with names for multi-SED comparison.")
    print("  (Double-click region \u2192 Properties \u2192 Text)")
    print("  Move, resize, or reshape regions \u2014 the plot updates automatically.")
    print("  Close the plot window to exit.")
    print("=" * 50)

    # Setup extractor
    extractor = SEDExtractor(
        frames=valid,
        coord_system=args.coord_system,
        bg_annulus_factor=bg_factor,
        flux_unit=args.flux_unit,
    )

    # Wrap extractor for single-region mode
    if args.single:
        original_extract = extractor.extract_all
        def extract_first(region_text):
            results = original_extract(region_text)
            return results[:1] if results else []
        extractor.extract_all = extract_first

    plotter = SEDPlotter(
        ds9=ds9,
        extractor=extractor,
        coord_system=args.coord_system,
        log_scale=args.log,
        poll_interval=args.poll_interval,
        save_csv=args.save_csv,
    )

    plotter.run()


if __name__ == '__main__':
    main()
