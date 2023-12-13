"""Matplotlib markers for truss."""
from matplotlib.path import Path


def get_roller_pin_support():
    """Roller pin marker."""
    verts = [
        (0, 0),
        (-3, -4),
        (3, -4),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
    ]
    path = Path.make_compound_path(
        Path(verts, codes),
        Path.circle((-2, -5), 1),
        Path.circle((2, -5), 1),
        Path.circle((0, -5), 1),
    )
    return path


def get_fixed_support():
    """Fixed support marker."""
    verts = [
        (0, 0),
        (3, 0),
        (3, -6),
        (-3, -6),
        (-3, 0),
        (0, 0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
    ]
    path = Path(verts, codes)
    return path


def get_pin_support():
    """Pin marker."""
    verts = [
        (0, 0),
        (-3, -4),
        (-3, -6),
        (3, -6),
        (3, -4),
        (0, 0)
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
    ]
    path = Path(verts, codes)
    return path


def get_track_pin_support():
    """Roller pin marker."""
    verts = [
        (0, 0),
        (-3, -4),
        (3, -4),
        (0, 0),
        (3, -5.5),
        (3, -6),
        (-3, -6),
        (-3, -5.5),
        (3, -5.5),

    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    return path


def get_track_fixed_support():
    """Roller pin marker."""
    verts = [
        (0, 0),
        (3, 0),
        (3, -4),
        (-3, -4),
        (-3, 0),
        (0, 0),
        (3, -5.5),
        (3, -6),
        (-3, -6),
        (-3, -5.5),
        (3, -5.5),

    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    return path


TF = get_track_fixed_support()
TP = get_track_pin_support()
RP = get_roller_pin_support()
P = get_pin_support()
F = get_fixed_support()
