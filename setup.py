import re

from setuptools import find_packages, setup

org = "PEtab-dev"
repo = "petab_select"


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative petab github links by absolute links."""
    raw_base = f"(https://raw.githubusercontent.com/{org}/{repo}/main/"
    embedded_base = f"(https://github.com/{org}/{repo}/tree/main/"
    # iterate over links
    for var in re.findall(r"\[.*?\]\((?!http).*?\)", txt):
        if re.match(r".*?.(png|svg)\)", var):
            # link to raw file
            rep = var.replace("(", raw_base)
        else:
            # link to github embedded file
            rep = var.replace("(", embedded_base)
        txt = txt.replace(var, rep)
    return txt


# project metadata
# noinspection PyUnresolvedReferences
setup(
    long_description=absolute_links(read("README.md")),
    long_description_content_type="text/markdown",
    url=f"https://github.com/{org}/{repo}",
    packages=find_packages(exclude=["doc*", "test*"]),
    include_package_data=True,
)
