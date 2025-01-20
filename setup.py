from setuptools import find_packages,setup
from typing import List
val = '-e.'
def get_requirments(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments= file_obj.readlines()
        requirments=[req.replace("\n","") for req in requirments]

        if val in requirments:
            requirments.remove(val)

setup(
    name='ML_Project',
    version='0.0.1',
    author="Sourav Sharma",
    author_email="souravbgp2210@gmail.com",
    packages=find_packages(),
    install_requirments = get_requirments('requirments.txt')
)