# Benchmarking out-of-reference detection

[![DOI](https://zenodo.org/badge/540036198.svg)](https://zenodo.org/badge/latestdoi/540036198)
[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/emdann/oor_benchmark/Test/main
[link-tests]: https://github.com/emdann/oor_benchmark/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/oor_benchmark

One of the goals of reference-based single-cell RNA-seq analysis is to detect altered cell states that are not observed in the reference dataset.
This repository contains code to benchmark workflows for integration and differential analysis on the task of detection of Out-of-reference (OOR) states, used in our [preprint](https://www.biorxiv.org/content/10.1101/2022.11.10.515939v1) on reference design for disease state identification. For code to reproduce analysis in the manuscript see the [reproducibility repo](https://github.com/MarioniLab/oor_design_reproducibility).

The structure of the API was inspired by the [OpenProblems](https://github.com/openproblems-bio/openproblems) task structure. This package was built using the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse).

## Getting started

-   [Quick-start example](https://nbviewer.org/github/emdann/oor_benchmark/blob/master/docs/notebooks/example.ipynb)

<!-- Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api]. -->

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install oor_benchmark:

1. Create a new conda environment

```bash
conda create --name oor-benchmark-env python=3.10
conda activate oor-benchmark-env
```

2. Install R and R dependencies

```bash
conda install conda-forge::r-base==4.0.5 bioconda::bioconductor-edger==3.32.1 conda-forge::r-statmod==1.4.37
```

<!--
1) Install the latest release of `oor_benchmark` from `PyPI <https://pypi.org/project/oor_benchmark/>`_:

```bash
pip install oor_benchmark
```
-->

3. Install the latest development version:

```bash
pip install git+https://github.com/emdann/oor_benchmark.git@master
```

<!-- ## Release notes

See the [changelog][changelog]. -->

<!-- ## Contact

Emma Dann <ed6@sanger.ac.uk> -->

<!-- For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker]. -->

## Citation

> Dann E., Teichmann S.A. and Marioni J.C. Precise identification of cell states altered in disease with healthy single-cell references. _biorXiv_ https://doi.org/10.1101/2022.11.10.515939

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/emdann/oor_benchmark/issues
[changelog]: https://oor_benchmark.readthedocs.io/latest/changelog.html
[link-docs]: https://oor_benchmark.readthedocs.io
[link-api]: https://oor_benchmark.readthedocs.io/latest/api.html
