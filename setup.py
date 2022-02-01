import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="clipcoders",
    py_modules=["clipcoders"],
    version="0.1",
    description="Wrappers for encoder/decoder applications of CLIP",
    author="Nam Hee Gordon Kim",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)