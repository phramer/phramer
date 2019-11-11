from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='phramer',
    version='0.1',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    entry_points={
        'console_scripts': [
            'phramer-preprocess = phramer.preprocess_text:main',
            # 'fairseq-validate = fairseq_cli.validate:cli_main',
        ],
    },
)

