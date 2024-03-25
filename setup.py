from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements()->List[str]:
    requirements:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            requirements = file.readlines()
            requirements = [req.replace('\n', '') for req in requirements]

            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)

    except FileNotFoundError:
        print('Requirements.txt file not found')

    return requirements


setup(
    name='dispute_segmentation',
    version='0.0.1',
    author='Aastha Tanwar',
    author_email='__@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
        
