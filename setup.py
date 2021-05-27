from setuptools import find_packages, setup

my_pckg = find_packages(include=["skbel"])

setup(
    name="skbel",
    version="1.0.1",
    packages=my_pckg,
    include_package_data=True,
    url="https://github.com/robinthibaut/skbel",
    license="MIT",
    author="Robin Thibaut",
    author_email="robin.thibaut@UGent.be",
    description="SKBEL - Bayesian Evidential Learning framework built on top of scikit-learn.",
    long_description="SKBEL - Bayesian Evidential Learning framework built on top of scikit-learn.",
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "loguru", "scikit-learn", "joblib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
