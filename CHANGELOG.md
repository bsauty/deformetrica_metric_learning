# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased
- new API: deformetrica can now be instantiated from python


## [4.0.0] - 2018-06-14
### Added
- Bugfix: version file not found. issue #24
- Easy install with `conda install -c pytorch -c conda-forge -c anaconda -c aramislab deformetrica`, without any manual compilation. 
- All existing deformetrica functionalities now work with 2d or 3d gray level images. 
- A L-BFGS optimization method can now be used for registration, regression, deterministic and bayesian atlases.
- Gradients are now automagically computed using PyTorch's autograd.
- It is now possible to perform all computations on the gpu through the `use-cuda` option. 

### Changed
- C++ is replaced by Python.
- The "exact" kernel is now named "torch"; the "cudaexact" kernel is now named "keops".
- The "deformable-object-type" xml entry is now split in two entries: "deformable-object-type" and "attachment-type". With this renamming, "NonOrientedSurfaceMesh" becomes a "SurfaceMesh" with a "Varifold" attachment (and an "OrientedSurfaceMesh" a "SurfaceMesh" with a "Current" attachment).

### Removed
- The Nesterov scheme for the gradient ascent optimizer (which was named "FastGradientAscent") is not available anymore. L-BFGS is more efficient though!

