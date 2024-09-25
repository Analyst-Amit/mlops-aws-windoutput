from setuptools import find_packages, setup

setup(
    name='your_project_name',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version='0.1.0',  # Update with your project's version
    description='Demo repository implementing an end-to-end MLOps workflow. Derived from a basic Python template.',
    author='Amit Gupta',  # Update with your name or organization
    author_email='amitgupta2533@gmail.com',  # Update with your email
    url='https://github.com/yourusername/your-repo',  # Update with your repository URL
)
