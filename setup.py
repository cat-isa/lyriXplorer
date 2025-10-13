#!/usr/bin/env python3
"""
Setup script for LyriXplorer
"""

import os
import sys
from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lyrixplorer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A personal lyrics-based song discovery app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lyrixplorer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "lyrixplorer=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 