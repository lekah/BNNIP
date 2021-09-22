from setuptools import setup, find_packages

setup(
    name="bnnip",
    version="0.0.1",
    description="",
    download_url="",
    author="Leonid Kahle",
    python_requires=">=3.6",
    packages=find_packages(include=["bnnip", "bnnip.*"]),
    zip_safe=True,
    scripts=['scripts/run_nequip_hmc.py']
)
