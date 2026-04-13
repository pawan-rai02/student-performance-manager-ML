from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path : str) -> List[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path, 'r') as f:
        requirements = [line.strip() for line in f]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Pawan',
    author_email= 'pawanvirat32@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)