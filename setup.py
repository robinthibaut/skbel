from setuptools import find_packages, setup

my_pckg = find_packages(include=["skbel"])
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
setup(
    name="skbel",
    version="2.1.13",
    packages=my_pckg,
    include_package_data=True,
    url="https://github.com/robinthibaut/skbel",
    license="BSD-3",
    author="Robin Thibaut",
    author_email="robin.thibaut@UGent.be",
    description="A set of Python modules to implement the Bayesian Evidential Learning (BEL) framework",
    long_description=LONG_DESCRIPTION,
    install_requires=["numpy",
                      "pandas",
                      "scipy",
                      "matplotlib",
                      "scikit-learn",
                      "scikit-image",
                      "joblib",
                      "pytest"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires=">=3.7",
)
