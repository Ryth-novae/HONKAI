# src/__init__.py

"""
N2HP Pipeline Source Package
============================

This package contains all core modules for the N2H+ molecular cloud core analysis pipeline.

Modules:
    loader:        Handles data loading, FITS header processing, and RMS calculation.
    processor:     Handles ScousePy and ACORNS clustering analysis.
    analyzer:      Handles physical property calculations (mass, density, Jeans length, virial mass, etc.).
    visualizer:    Handles all plotting functions (3D plots, dendrograms, statistics, contours).
    utils:         Contains common mathematical formulas, geometry calculations, and utility functions.
"""

# Explicitly export submodules for convenient external use
from . import loader
from . import processor
from . import analyzer
from . import visualizer
from . import utils
from . import multi_wavelength_analyzer

# Define behavior for 'from src import *'
__all__ = [
    'loader',
    'processor',
    'analyzer',
    'visualizer',
    'utils'
]