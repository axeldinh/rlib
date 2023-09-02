
import pathlib
from setuptools import setup

import subprocess
import os

path = os.path.dirname(os.path.abspath(__file__))

# Get the remote url
remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], 
                                        cwd=path).decode("utf-8").strip()

# Get the commit hash
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                        cwd=path).decode("utf-8").strip()

# Get the commit message
commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"],
                                            cwd=path).decode("utf-8").strip().replace("|", " ").replace("\n", " ")

# Get the branch name
branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                        cwd=path).decode("utf-8").strip()

with open(os.path.abspath("src/rlib/__git_infos__.py"), "w") as infos_file:
    infos_file.write(f'__remote_url__ = "{remote_url}"\n')
    infos_file.write(f'__commit_hash__ = "{commit_hash}"\n')
    infos_file.write(f'__commit_message__ = "{commit_message}"\n')
    infos_file.write(f'__branch_name__ = "{branch_name}"\n')

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
