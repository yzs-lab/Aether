from setuptools import find_packages, setup


setup(
    name="aether",
    version="0.1.0",
    description="Aether: simulation and measurement tooling for GreenServe intelligence-per-watt experiments.",
    author="Aether Contributors",
    author_email="aether@yezhisheng.me",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["PyYAML>=5.4"],
    extras_require={"dev": ["pytest>=8.3"], "sglang": ["pynvml>=11.5"]},
    entry_points={"console_scripts": ["aether=aether.cli:main"]},
    python_requires=">=3.13",
)
