import flurs

VERSION = flurs.__version__

DISTNAME = "flurs"
DESCRIPTION = "A library for streaming recommendation algorithms"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
AUTHOR = "Takuya Kitazawa"
AUTHOR_EMAIL = "k.takuti@gmail.com"
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = "MIT"
URL = "https://github.com/takuti/flurs"
DOWNLOAD_URL = "https://pypi.org/project/flurs/#files"


def setup_package():
    from setuptools import setup, find_packages

    metadata = dict(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Development Status :: 4 - Beta",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        packages=find_packages(exclude=["*tests*"]),
        install_requires=[
            "numpy>=1.14.6",
            "scipy>=1.6.3",
            "scikit_learn>=1.0",
            "mmh3>=3.0.0",
        ],
        extras_require={
            "docs": [
                "sphinx_rtd_theme",
                "sphinx-gallery",
            ],
            "tests": {
                "pytest>=5.0.1",
                # https://github.com/pytest-dev/pytest/issues/4608
                "pytest-remotedata==0.3.2",
                "flake8>=3.8.2",
                "black>=21.6b0",
                "mypy>=0.770",
                "pre-commit",
            },
        },
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
