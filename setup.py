from distutils.core import setup

setup(
    name='scvis',
    version='0.1.0',
    description='Modeling and visualizing high-dimensional single-cell data',
    author='Jiarui Ding',
    author_email='jiarui.ding@gmail.com',
    url='http://compbio.bccrc.ca',
    package_dir={'': 'lib'},
    packages=['scvis'],
    package_data={'': ['config/model_config.yaml']},
    scripts=['scvis'],
    install_requires=[
        'tensorflow >= 1.1',
        'matplotlib >= 1.5.1',
        'numpy >= 1.11.1',
        'PyYaml >= 3.11',
        'pandas >= 0.19.1']
)
