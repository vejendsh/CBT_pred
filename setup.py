from setuptools import setup, find_packages

setup(
    name="cbt_pred",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "ansys-fluent-core",
        "numpy",
        "pandas",
    ],
) 