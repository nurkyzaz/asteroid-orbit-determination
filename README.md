# Asteroid Orbit Determination (Angles-Only Gauss OD) + Astrometry/Photometry Utilities (Python)
### Computational Astrophysics Project — Summer Science Program in Astrophysics (SSP)

This repository contains a Python implementation of **angles-only orbit determination** for a Solar System object using the **Method of Gauss** (three observations of RA/Dec + times) with initialization via the **Scalar Equation of Lagrange** and iterative refinement using **f and g functions**. The code also includes supporting utilities for **centroiding**, **aperture photometry**, and **linear plate reduction (LSPR)**.

The main output of the orbit determination pipeline is the object’s heliocentric **state vector** at the middle observation \((\mathbf{r}_2, \dot{\mathbf{r}}_2)\) and corresponding **orbital elements** \((a, e, i, \Omega, \omega, M)\).

The research and code were developed as part of the **Summer Science Program in Astrophysics (SSP)** conducted at the **University of North Carolina at Chapel Hill**, where the goal was to determine the heliocentric orbit of a Near-Earth asteroid 1985 JA from observations.

---

## What this code does 

Given **three observations** (time, RA, Dec) and the corresponding **Earth→Sun vectors** \(\mathbf{R}_i\) in AU at those times, the pipeline:

1. Converts observation timestamps to **Julian Dates**
2. Converts each RA/Dec to a unit line-of-sight vector \(\hat{\rho}_i\)
3. Computes Gauss “D-constants” using dot/cross products of \(\hat{\rho}_i\) and \(\mathbf{R}_i\)
4. Uses the **Scalar Equation of Lagrange** to generate candidate values of \(|\mathbf{r}_2|\) (can be multiple positive real roots)
5. Performs a **first-pass** Gauss solution using truncated series for **f and g**
6. Iteratively refines the solution by updating:
   - ranges \(\rho_1, \rho_2, \rho_3\)
   - heliocentric vectors \(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3\)
   - velocity \(\dot{\mathbf{r}}_2\)
   - **light-travel-time correction**
   - improved f and g via a closed-form Kepler solution (Newton’s method for \(\Delta E\))
7. Rotates the final vectors from ecliptic to equatorial frame and computes orbital elements:
   - semi-major axis \(a\)
   - eccentricity \(e\)
   - inclination \(i\)
   - longitude of ascending node \(\Omega\)
   - argument of periapsis \(\omega\)
   - mean anomaly \(M\) (at the middle observation and at a chosen epoch)

---

## Repository structure

- **`OD_nurkyz_final.py`**
  - Main driver script for orbit determination.
  - Reads an input file of observations, computes candidate Lagrange roots, performs iterative OD, and prints the final orbital elements.

- **`odlib_nurkyz.py`**
  - Library of reusable functions including:
    - Orbit determination core: `lagrange()`, `fg()`, `final_elements()`, plus helpers for unit vectors / conversions
    - Image/measurement utilities: `centroid()`, `photometry()`
    - Astrometry: `LSPR()` (linear plate solution)

---

## Requirements

Python 3.9+ recommended.

Dependencies:
- `numpy`
- `scipy`
- `astropy` (for FITS I/O used by centroid/photometry utilities)
- `matplotlib` (optional; used for plotting in some workflows)

Install:
```bash
pip install numpy scipy astropy matplotlib
