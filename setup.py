import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="m2oe",
    version="0.0.1",
    author="Fabrizio Angiulli, Fabio Fassetti, Simona Nisticò, Luigi Palopoli",
    author_email="simona.nistico@dimes.unical.it",
    description="Masking Models for Outlier Explanation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AIDALab-DIMES/M2OE",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'mlxtend==0.22.0',
        'tqdm',
        'numpy',
        'pandas',
        'scikit_learn>=1.0.2',
        'tensorflow>=2.3.0',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)