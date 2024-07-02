from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line and not line.startswith('#')]

setup(
    name='RLAlgorithms',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Your Name',
    author_email='roihezkiyahu@gmail.com',
    description='A collection of reinforcement learning algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/roihezkiyahu/RLAlgorithms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
