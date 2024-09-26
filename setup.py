from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = "mlops-aws-windoutput"
DESCRIPTION = "Example regression model package from Train In Data."
EMAIL = "amitgupta2533@gmail.com"
AUTHOR = "Amit Gupta"
REQUIRES_PYTHON = ">=3.10.12"

# Load the package's VERSION file if you have one
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "src"

# # Function to list requirements
# def list_reqs(fname="requirements.txt"):
#     requirements_path = ROOT_DIR / fname  # Point to the root requirements.txt
#     with open(requirements_path) as fd:
#         return fd.read().splitlines()

# The actual setup function
setup(
    name=NAME,
    version="0.1.0",  # Set a default version or load from a VERSION file
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # install_requires=list_reqs(),
    include_package_data=True,
)
