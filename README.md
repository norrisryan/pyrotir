# ROTIR JAX: Python/JAX Port

Python/JAX implementation of ROTIR for stellar surface reconstruction from optical interferometry data.

## Overview

ROTIR is a package for reconstructing stellar surface brightness maps from optical interferometry measurements. This Python/JAX port provides:

- **GPU Acceleration**: JAX enables efficient computation on GPUs
- **Modern Optimization**: Integration with modern optimization libraries
- **Extensibility**: Clean Python API for future NIFTy8 integration

## Project Status

**Current Phase**: Step 1 - Data Structures (Complete)

This is an active port following the detailed specification in `rotir_python_impementatio_spec.pdf`.

### Implementation Progress

- [x] Step 1: Data structures (datatypes.py)
- [ ] Step 2: HEALPix tessellation
- [ ] Step 3: Base geometry (rotation, projection)
- [ ] Step 4: Polygon Fourier transform
- [ ] Step 5: Observables and chi-squared
- [ ] Step 6: OIFITS I/O
- [ ] Step 7: Integration testing
- [ ] Step 8: Rapid rotator geometry
- [ ] Step 9: Roche lobe geometry and orbits
- [ ] Step 10: Regularization
- [ ] Step 11: Reconstruction optimizer
- [ ] Step 12: Multi-epoch support

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pyrotir

# Install dependencies (CPU version)
pip install -e .

# Or for GPU support
pip install -e ".[gpu]"

# For development
pip install -e ".[dev]"
```

## Project Structure

```
rotir_jax/
├── __init__.py
├── datatypes.py              # Core data structures
├── tessellation/
│   └── healpix.py            # HEALPix tessellation
├── geometry/
│   ├── base.py               # Rotation, projection, visibility
│   ├── rapid_rotator.py      # Rapid rotator models
│   ├── roche.py              # Roche lobe geometry
│   └── orbits.py             # Binary orbit computation
├── forward_model/
│   ├── polyft.py             # Polygon Fourier transform
│   └── observables.py        # Compute V², T3, chi²
├── io/
│   └── oifits_reader.py      # OIFITS file loading
├── regularization/
│   └── regularizers.py       # TV, L2, MEM regularization
├── reconstruction/
│   └── optimizer.py          # L-BFGS-B optimization
└── tests/
    └── ...                   # Unit and integration tests
```

## Reference

This is a port of the original Julia implementation. See the Julia source files (.jl) in the repository root for the reference implementation.

## License

See LICENSE file for details.
