from setuptools import find_packages, setup

setup(
    name='mach_1_functions',
    packages=find_packages(include=['mach_1_functions']),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn"
    ],
    version='0.1.3',
    description='Mach-1 Analysis Functions',
    author='Renato Castillo',
    author_email="castillo.renato@biomomentum.com"
)