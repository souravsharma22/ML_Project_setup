from setuptools import find_packages, setup
from typing import List

val = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if val in requirements:
            requirements.remove(val)

    return requirements 

setup(
    name='ML_Project',
    version='0.0.1',
    author="Sourav Sharma",
    author_email="souravbgp2210@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')  
)
