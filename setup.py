from setuptools import setup, find_packages

setup(
    name="gpuperf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "pyyaml>=6.0",
    ],
    author="GPUPerf-3B Team",
    description="A Profiler-Grounded GPU Optimization Model",
)

