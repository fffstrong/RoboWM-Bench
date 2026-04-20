import setuptools

setuptools.setup(
    name="phantom",
    version="0.1",
    packages=setuptools.find_packages(exclude=["submodules", "submodules.*"]),
)