import setuptools


setuptools.setup(
    name="tfdlg",
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow>=2.3,<3.0",
        "sentencepiece==0.1.91",
        "scipy>=1.5.0,<2.0",
        "pydantic>=1.6.1,<2.0",
    ],
    extras_require={
        "test": ["pytest>=5.0,<60", "black>=20.8b1,<21"],
    },
    version="0.1.0",
    author="Noriyuki Abe",
)
