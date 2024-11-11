from setuptools import find_packages, setup

setup(
    name='mach_1_functions',
    packages=find_packages(include=['mach_1_functions']),
    install_requires=[
        "numpy",
        "pandas",
        "scipy"
    ],
    version='0.1.2',
    description='Mach-1 Analysis Functions',
    author='Renato Castillo',
    author_email="castillo.renato@biomomentum.com"
)