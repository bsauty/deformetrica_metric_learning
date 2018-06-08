# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.0-rc] - 2018-06-08
### Added
- All existing deformetrica functionnalities now work with 2d or 3d gray level images.
- A L-BFGS optimization method can now be used for registration, regression, deterministic and bayesian atlases.
- A C++/Cuda kernel is now available: [Keops](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops).
- Gradients are now automagically computed using PyTorch's autograd.

### Changed
- C++ is replaced by Python.
- The "exact" kernel is now named "torch"; the "cudaexact" kernel is now named "keops".
- The "deformable-object-type" xml entry is now split in two entries: "deformable-object-type" and "attachment-type". With this renamming, "NonOrientedSurfaceMssh" 
- Deformetrica CLI now uses argparse to manage user input.

### Removed
- The Nesterov scheme for the gradient ascent optimizer (which was named "FastGradientAscent")

