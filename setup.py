from setuptools import setup, find_packages

setup(
    name='kalmangrad',
    version='0.0.1',
    description="Automated, smooth, N'th order derivatives of non-uniformly sampled time series data",
    author='Hugo Hadfield',
    author_email='hadfield.hugo@gmail.com',
    url='https://github.com/hugohadfield/kalmangrad',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'bayesfilter',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
