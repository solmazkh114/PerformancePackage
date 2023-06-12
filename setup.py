from setuptools import setup


setup(
    name='performance',
    version='0.0.1',
    description='An easy way of extracting all performance metrics of an ML model',
    py_modules=["ml-performance"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9"
    ],
    install_requires=[
        "numpy",
        "pandas",
        "seaborn",
        "scikit-learn"
    ],
    author="Solmaz Khajehpour",
    author_email="solmazkh114@gmail.com"
)