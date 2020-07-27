from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

setup(
    name="tinyclues_fletcher",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Pandas ExtensionDType/Array backed by Apache Arrow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xhochy/fletcher",
    author="Uwe L. Korn",
    author_email="fletcher@uwekorn.com",
    license="MIT",
    classifiers=[  # Optional
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    install_requires=["pandas~=1.0.0", "pyarrow==0.17.1", "numba>=0.50.1", "six"],
)
