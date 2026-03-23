# GPkit

GPkit is a Python package for defining and solving geometric programming (GP) models.
It provides symbolic variables with physical units, composable constraint sets, and
interfaces to convex solvers — delivering reliable, globally-optimal solutions to
engineering design problems.

[![Test Status](https://github.com/beautifulmachines/gpkit-core/actions/workflows/tests.yml/badge.svg)](https://github.com/beautifulmachines/gpkit-core/actions/workflows/tests.yml)
[![Lint Status](https://github.com/beautifulmachines/gpkit-core/actions/workflows/lint.yml/badge.svg)](https://github.com/beautifulmachines/gpkit-core/actions/workflows/lint.yml)

## Installation

```bash
pip install gpkit-core
```

Supported solvers: [MOSEK](https://www.mosek.com) and [cvxopt](https://cvxopt.org)
(cvxopt is open source and installed by default).

## Documentation

See [docs/](docs/) for full documentation including getting started, modeling
conventions, and examples.

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
workflow details.

## Acknowledgments

Originally developed with Ned Burnell, whose extensive contributions were
foundational to the early design.
