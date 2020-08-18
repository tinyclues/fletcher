#!/bin/bash

set +x
set -eo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHON_VERSION=$1
export USE_DEV_WHEELS=$2
export CONDA_PKGS_DIRS=$HOME/.conda_packages
export MINICONDA=$HOME/miniconda
export MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
export PATH="$MINICONDA/bin:$PATH"

wget --no-verbose -O miniconda.sh $MINICONDA_URL
bash miniconda.sh -b -p $MINICONDA
export PATH="$MINICONDA/bin:$PATH"

conda config --set auto_update_conda false
conda config --add channels conda-forge
conda install -y mamba yq jq

conda create -y -q -c conda-forge -n fletcher \
    python=${PYTHON_VERSION} \
    numba \
    "pandas>=1.0" \
    "pyarrow>=0.17.1" \
    six \
    pytest \
    pytest-cov \
    hypothesis \
    setuptools_scm \
    pip \
    codecov \
    sphinx \
    pre_commit \
    dask \
    numpydoc \
    sphinx \
    sphinx_rtd_theme \
    sphinxcontrib-apidoc

source activate fletcher

if [ "${PYTHON_VERSION}" = "3.7" ]; then
  pre-commit install
  pre-commit run -a
fi

if [ "${USE_DEV_WHEELS}" = "nightlies" ]; then
    echo "Installing Pandas dev"
    conda uninstall -y --force pandas
    PRE_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS pandas
fi

python -m pip install -e . --no-build-isolation --no-use-pep517
py.test --junitxml=test-reports/junit.xml --cov=./ --cov-report=xml

# Do a second run with JIT disabled to produce coverage and check that the
# code works also as expected in Python.
if [ "${PYTHON_VERSION}" = "3.6" ]; then
  # These don't work with Python 2.7 as it supports less operators than 3.6
  NUMBA_DISABLE_JIT=1 py.test --junitxml=test-reports/junit.xml --cov=./ --cov-report=xml
fi

# Check documentation build only in one job
if [ "${PYTHON_VERSION}" = "3.7" ]; then
  pushd docs
  make html
  popd
fi
