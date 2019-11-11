from setuptools import setup, find_packages
from os.path import join, dirname


requirements = ['pandas==0.24.2',
                'dvc==0.66.11',
                'comet-ml==2.0.16',
                'bs4==0.0.1',
                'tqdm==4.37.0',
                'google-cloud-storage==1.19.0']

dev_requirements = ['pre-commit',
                    'black']

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
    install_requires=requirements,
    extras_require={'dev': dev_requirements},
)

