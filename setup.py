import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jupyter-psychoacoustics",
    version="0.0.1",
    author="Ben Willmore",
    author_email="ben@willmore.eu",
    description="Psychoacoustics using jupyter / Google Colab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beniamino38/jupyter-psychoacoustics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
