
import pathlib
from setuptools import setup

from src.rlib.utils import get_git_infos

get_git_infos()

def get_requirements():
    with open(pathlib.Path(__file__).parent.resolve().joinpath("requirements.txt")) as requirements_file:
        return requirements_file.read().splitlines()
    
def get_version():
    with open(pathlib.Path(__file__).parent.resolve().joinpath("src/rlib/__init__.py")) as version_file:
        for line in version_file.readlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')

def get_long_description():
    with open(pathlib.Path(__file__).parent.resolve().joinpath("README.md")) as readme_file:
        return readme_file.read()
    
setup(
    name="rlib",
    version=get_version(),
    description="A library for Reinforcement Learning",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Axel Dinh Van Chi",
    author_email="axeldvc@gmail.com",
    package_data={
        "rlib.envs.flappy_bird_gymnasium": ["assets/*"]
    },
    install_requires=get_requirements(),
)
