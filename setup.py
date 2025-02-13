from setuptools import setup

setup(
    name='optimal_cocluster',
    version='0.1.1',
    packages=['optimal_cocluster'],
    url='https://github.com/JohannMue/Optimal_Coclust',
    license='GPL-3.0',
    author='J',
    author_email='',
    description='This package provides simple tools to evaulate coclustering solutions. The package is made to work with the CoClust-Package',
    setup_requires=["numpy"],
    install_requires=['numpy', 'coclust'],
)
