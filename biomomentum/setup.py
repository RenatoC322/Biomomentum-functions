from setuptools import find_packages, setup

setup(
    name='biomomentum',
    packages=find_packages(include=['biomomentum']),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn"
    ],
    version='0.1.2',
    description='Mach-1 Analysis Functions',
    author='Renato Castillo',
    author_email="castillo.renato@biomomentum.com"
)