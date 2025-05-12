from pathlib import Path
from setuptools import find_packages, setup

# read the README.md for the long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="biomomentum",
    version="0.1.7",                        
    description="Mach-1 Analysis Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Renato Castillo",
    author_email="castillo.renato@biomomentum.com",
    url="https://github.com/RenatoC322/Biomomentum-functions/tree/main",  # your repo URL

    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "opencv-python",
    ],

    packages=find_packages(include=["biomomentum", "biomomentum.*"]),
    include_package_data=True,   # respect MANIFEST.in

    package_data={
        "biomomentum.data": ["*.mat"],  # include all .mat in data/
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)