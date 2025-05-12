from setuptools import find_packages, setup

setup(
    name='biomomentum',
    packages=find_packages(include=['biomomentum']),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "opencv-python"
    ],
    version='0.1.5',
    description='Mach-1 Analysis Functions',
    author='Renato Castillo',
    author_email="castillo.renato@biomomentum.com",
    python_requires=">=3.11",
)