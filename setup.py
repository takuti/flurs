from setuptools import setup


setup(
    name='orec',
    version='0.0.1',
    description='A library for top-k online recommendation from implicit feedback',
    author='Takuya Kitazawa',
    author_email='k.takuti@gmail.com',
    license='MIT',
    url='https://github.com/takuti/orec',
    packages=['orec'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit_learn'],
)
