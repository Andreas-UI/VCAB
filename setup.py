from setuptools import setup, find_packages
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="vcab",
    version="v0.1.0-beta",
    description="""A tool to classify autism behaviour in children using deep learning model""",
    long_description_content_type="text/markdown",
    author="Andreas Susanto",
    packages=find_packages(include=["vcab"]),
    package_data={'': ['*.ckpt']},
    install_requires=required,
    license="MIT",
    url="https://github.com/Andreas-UI/VCAB",
    python_requires=">=3.10",
)
