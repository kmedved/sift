import pathlib
import re

from setuptools import setup


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

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='sift',
    version=read_version(),
    description='minimum-Redundancy-Maximum-Relevance algorithm for feature selection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smazzanti/mrmr',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='MIT',
    packages=['sift'],
    install_requires=[
        'tqdm',
        'joblib',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        'categorical': ['category_encoders'],
        'polars': ['polars>=0.12.5', 'pyarrow'],
        'numba': ['numba'],
        'test': ['pytest'],
        'all': [
            'category_encoders',
            'polars>=0.12.5',
            'pyarrow',
            'numba',
        ],
    },
    zip_safe=False
)
