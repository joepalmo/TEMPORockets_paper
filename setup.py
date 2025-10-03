from setuptools import setup, find_packages

setup(
    name="rocketplumes",
    version="0.1.0",
    packages=find_packages(),
    author="Joe Palmo",
    description="TEMPO NO2 plume detection and analysis for rocket launches",
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scikit-image",
        "cartopy",
        "geopandas",
        "rasterio",
        "pyproj",
        "shapely",
        "requests",
        "dask",
        "distributed",
        "dask-jobqueue",
        "netcdf4",
        "h5netcdf",
        "h5py",
        "jupyterlab",
        "ipykernel",
        "tqdm",
        "joblib"
    ],
    python_requires=">=3.10"
)
