import re
import setuptools
import sys


if not sys.version_info >= (3, 5):
    exit("Sorry, Python must be later than 3.5.")


setuptools.setup(
    name="tensorflow-font2char2word2sent2doc",
    version=re.search(
        r'__version__ *= *"([0-9]+\.[0-9]+\.[0-9]+)" *\n',
        open("font2char2word2sent2doc/__init__.py").read()).group(1),
    description="TensorFlow implementation of Hierarchical Attention Networks "
                "for Document Classification",
    long_description=open("README.md").read(),
    license="Public Domain",
    author="Yota Toyama",
    author_email="raviqqe@gmail.com",
    url="https://github.com/raviqqe/tensorflow-font2char2word2sent2doc/",
    packages=["font2char2word2sent2doc"],
    install_requires=[
        "argtyp",
        "char2image",
        "tensorflow-qnd<=1",
        "tensorflow-extenteten<=1",
        "tensorflow-qndex<=1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Public Domain",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
    ],
)
