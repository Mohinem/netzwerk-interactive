from setuptools import setup, find_packages

setup(
    name="custom-gym-environments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.27.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",  # For rendering
    ],
    author="Your Name",
    description="Collection of custom Gymnasium environments for reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/custom-gym-environments",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
