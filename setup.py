"""Editable install for Conf-Reranker."""

from setuptools import find_packages, setup

setup(
    name="conf-reranker",
    version="0.1.0",
    description="Confidence-propagating cross-encoder reranking for EDA RAG.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Anonymous Authors",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "scripts*"]),
    install_requires=[
        "torch>=2.0,<3.0",
        "transformers>=4.38,<5.0",
        "tokenizers>=0.15",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
