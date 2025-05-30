from setuptools import setup, find_packages

setup(
    name="pycodex",
    version="0.1.9",
    description="Co-detection by indexing (CODEX) analysis using Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="wuwenrui555",
    author_email="wuwenruiwwr@outlook.com",
    url="https://github.com/SizunJiangLab/pycodex",
    packages=find_packages(),
    install_requires=["deepcell", "rasterio", "numpy<=1.26.4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
