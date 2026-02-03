# Asteroid Orbit Determination via Classical Methods

This project implements classical orbit determination for near-Earth objects
using astrometric observations and numerical methods. The goal is to recover
heliocentric position, velocity, and orbital elements from three observations.

## Methods
- Julian Date conversion and time normalization
- Line-of-sight unit vector construction from RA/Dec
- Gauss–Lagrange scalar equation for heliocentric distance
- Polynomial root finding for admissible solutions
- Iterative refinement using Newton–Raphson methods
- Light travel time correction
- Coordinate rotations between equatorial and ecliptic frames
- Orbital element computation (a, e, i, Ω, ω, M)

## Code Structure
- `odlib_nurkyz.py`: reusable library of orbital mechanics and photometry functions
- `OD_nurkyz_final.py`: main pipeline for orbit determination and iteration

## Results
The algorithm converges to physically meaningful orbital elements within a
specified tolerance and reproduces reference ephemerides with small relative error.

## Skills Demonstrated
- Numerical methods
- Scientific Python
- Computational astrophysics
- Vector calculus and coordinate transformations

## Future Work
- Batch processing of multiple objects
- Integration with survey data pipelines
- Uncertainty propagation
