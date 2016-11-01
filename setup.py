from setuptools import setup


setup(
    name='flurs',
    version='0.0.1',
    description='A library for streaming recommender systems',
    author='Takuya Kitazawa',
    author_email='k.takuti@gmail.com',
    license='MIT',
    url='https://github.com/takuti/flurs',
    packages=['flurs'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit_learn'],
)
