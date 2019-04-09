from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here,"README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here,"bbx","VERSION"),"r") as f:
    __version__ = f.read().strip()

setup(name = "bbx",
    version = __version__,
    description = "Simple bounding box operations",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = "Roman Juranek",
    author_email = "rjuranek1983@gmail.com",
    keywords = "bounding box, non maxima suppression",
    packages = ["bbx"],
    python_requires = ">=2.6",
    install_requires=["numpy"],
    include_package_data=True,
)
