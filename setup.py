from setuptools import find_packages, setup

setup(
    name="credit-risk-intelligence-platform",
    version="0.1.0",
    description="End-to-end credit default risk prediction platform with explainability, MLOps, and drift monitoring.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
