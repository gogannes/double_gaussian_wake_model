from setuptools import setup

REQUIRED = [
    "matplotlib",
    "numpy",
    "pytest",
    "scipy",
]

setup(
    name='double_gaussian_wake_model',
    version='2.1',
    py_modules=['double_gaussian_wake_model'],
    url='https://github.com/gogannes/double_gaussian_wake_model',
    license='BSD-2-Clause',
    author='Johannes Schreiber',
    author_email='gogannes@gmail.com',
    description='double gaussian wind turbine wake model as published in Schreiber, J., Balbaa, A., and Bottasso, C. L.: Brief communication: A double-Gaussian wake model, Wind Energ. Sci., 5, 237–244, https://doi.org/10.5194/wes-5-237-2020, 2020',
    python_requires=">=3.10",
    install_requires=REQUIRED
)
