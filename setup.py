"""FluRS
============

FluRS is a Python library for online item recommendation. The name indicates Flu-* (Flux, Fluid, Fluent) recommender systems which incrementally adapt to dynamic user-item interactions in a streaming environment.

"""
import flurs
VERSION = flurs.__version__

DISTNAME = 'flurs'
DESCRIPTION = 'A library for streaming recommendation algorithms'
LONG_DESCRIPTION = __doc__ or ''
AUTHOR = 'Takuya Kitazawa'
AUTHOR_EMAIL = 'k.takuti@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = 'MIT'
URL = 'https://github.com/takuti/flurs'
DOWNLOAD_URL = 'https://pypi.org/project/flurs/#files'


def setup_package():
    from setuptools import setup, find_packages

    metadata = dict(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Unix',
                     'Operating System :: MacOS',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6'],
        packages=find_packages(exclude=['*tests*']),
        install_requires=[
            'numpy',
            'scipy',
            'scikit_learn',
            'mmh3'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
