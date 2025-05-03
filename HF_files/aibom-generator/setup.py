from setuptools import setup, find_packages

setup(
    name="aibom_generator",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "huggingface_hub",
        "transformers",
        "cyclonedx-python-lib",
        "requests",
        "pyyaml",
    ],
    python_requires=">=3.8",
)
