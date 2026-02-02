"""
Minimal setup.py shim for backward compatibility.

All package metadata is now defined in pyproject.toml.
This file exists only for compatibility with tools that don't yet support PEP 517/518.
"""

from setuptools import setup

setup()
