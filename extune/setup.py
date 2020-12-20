import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extune",
    version="0.0.1",
    author="Sumner Magruder",
    author_email="sumner.magruder@zmnh.uni-hamburg.de",
    description="Sacred File Observer utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/sfo/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
