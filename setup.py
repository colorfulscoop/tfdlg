import setuptools


setuptools.setup(
    name="tfchat",
    packages=setuptools.find_packages(),
    install_requires=[
        # Install tensorflow 2.X
        "tensorflow~=2.0",
        # "tensorflow_text~=2.0"
        # "tensorflow-hub",
        # Install Scikit-learn
        # "scikit-learn>=0.22"
        "pydantic==1.6.1",
    ],
    version="0.1.0",
    author="Noriyuki Abe",
)
