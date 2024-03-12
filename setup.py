from setuptools import setup, find_packages

setup(
    name='magplot',
    version='0.1',
    description='',
    url='https://github.com/mgjeon/magnetic_field_line',
    author='Mingyu Jeon',
    author_email='mgjeon@khu.ac.kr',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyvista',
        'matplotlib',
    ],
)