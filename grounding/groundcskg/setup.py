from setuptools import setup, find_packages


test_deps = ['pytest', 'pytest-cov', 'codecov']

setup(
    name="graphify",
    version="0.1.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "allennlp>=0.9.0",
        "numpy>=1.17.4",
        "spacy>=2.2.3",
        "tqdm>=4.40.2",
        "wordfreq>=2.2.1",
        "fasttext",
    ],
    dependency_links=['git+https://github.com/facebookresearch/fastText.git/@v0.9.2#egg=fasttext-0'],
    tests_require=test_deps,
    extras_require={
        'test': test_deps
    }
)
