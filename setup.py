from setuptools import setup, find_packages

setup(
    name="kernix",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "pyyaml>=6.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
    ],
    author="Sarim Ahsan",
    description="Kernix: A Profiler-Grounded GPU Kernel Optimization Engine",
    url="https://github.com/sarimahsan/Kernix",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
