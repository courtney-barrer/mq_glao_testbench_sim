"""
optics_simulator

Hybrid geometric ray-tracing and scalar wave-optics utilities for the
MQ GLAO / ULTIMATE testbench simulation.

The package is designed to keep the original beam_trace.py API backwards
compatible while adding a separate Fresnel/Wavefront2D layer.

Main modules
------------
fresnel
    Low-level scalar diffraction and propagation functions.

wavefront
    Wavefront2D dataclass and conversion helpers.

wave_optics
    High-level optical operations on Wavefront2D objects.

beam_trace_wave
    Bridge functions from existing beam_trace-style geometry/samples to
    Wavefront2D.

psf_tools
    PSF generation and basic analysis utilities.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]