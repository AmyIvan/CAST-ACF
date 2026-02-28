#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cast-acf",
    version="1.0.0",
    author="Yuming Ai",
    author_email="",
    description="CAST-ACF: Robust Generation and Evaluation for Multi-Granularity Timeline Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmyIvan/CAST-ACF",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "sentence-transformers>=2.2.0",
        "rank-bm25>=0.2.2",
        "jieba>=0.42.1",
        "rouge-chinese>=1.0.3",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "ujson>=5.8.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
)
