import pathlib
import re

from setuptools import setup, find_packages


def read_version():
    version_path = pathlib.Path(__file__).resolve().parent / "sift" / "__init__.py"
    match = re.search(
        r"^__version__\s*=\s*[\"']([^\"']+)[\"']",
        version_path.read_text(encoding="utf8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Unable to find __version__ in sift/__init__.py")
    return match.group(1)

try:
    with open("README.md", encoding="utf8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Feature selection toolbox"

setup(
    name='sift',
    version=read_version(),
    description='Feature selection toolbox (mRMR, JMI/JMIM, CEFS+, Stability Selection, ...)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smazzanti/mrmr',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='MIT',
    packages=find_packages(exclude=("tests*", "docs*", "examples*")),
    install_requires=[
        'numba',
        'joblib',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        'categorical': ['category_encoders'],
        'catboost': ['catboost'],
        'test': ['pytest'],
        'all': [
            'category_encoders',
            'catboost',
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False
)
