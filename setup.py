"""Q1K setup file.

Authors:
Christian O'Reilly <christian.oreilly@sc.edu>
James Desjardins <jim.a.desjardins@gmail.com>
License: MIT
"""
from pathlib import Path
from setuptools import setup, find_packages

with Path("requirements.txt").open() as f:
    requirements = f.read().splitlines()

extras = {
    "test": "requirements_testing.txt",
    "doc": "./docs/requirements_doc.txt",
}

extras_require = {}
for extra, req_file in extras.items():
    with Path(req_file).open() as file:
        requirements_extra = file.read().splitlines()
    extras_require[extra] = requirements_extra

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

#qc_entry_point = ["pylossless_qc=pylossless.dash.pylossless_qc:main"]
setup(
    name="q1k",
    version="0.0.1",
    description="Toolbox with utilitary tools for analyzing data from the"
                " Quebec 1000 families project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Christian O'Reilly",
    author_email="christian.oreilly@sc.edu",
    url="https://github.com/lina-usc/q1k",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    #entry_points={"console_scripts": qc_entry_point},
)