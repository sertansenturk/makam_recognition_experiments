import os
import re

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))
EXP_DIR = "src"


def get_version():
    """ Read version from __init__.py

    Raises:
        ValueError: if __init__ is not read, or __version__ is not in __init__

    Returns:
        str -- value of __version__ as defined in __init__.py
    """
    version_file2 = os.path.join(
        HERE, EXP_DIR, "experimentation_code", "__init__.py")
    with open(version_file2) as f:
        init_contents = f.read().strip()

        exp = r"^__version__ = ['\"]([^'\"]*)['\"]"
        mo = re.search(exp, init_contents, re.M)
        if mo:
            return mo.group(1)

        raise ValueError("Unable to find version string in %s." % (f,))


setup(
    name="experimentation_code",
    version=get_version(),
    author="Sertan Senturk",
    author_email="contact AT sertansenturk DOT com",
    maintainer="Sertan Senturk",
    maintainer_email="contact AT sertansenturk DOT com",
    url="https://github.com/sertansenturk/makam_recognition_experiments",
    description="Makam Recognition Experiments",
    download_url=(
        "https://github.com/sertansenturk/makam_recognition_experiments.git"
        if "dev" in get_version()
        else "https://github.com/sertansenturk/makam_recognition_experiments/"
        "releases/tag/v{0:s}".format(get_version())
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU Affero General Public License v3 or "
        "later (AGPLv3+)",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    platforms="Linux",
    license="agpl 3.0",
    keywords=(
        "audio-recordings "
        "makam-music "
        "mode-recognition "
        "machine-learning "
        "music-information-retrieval "
        "computational-analysis"
    ),
    packages=find_packages(EXP_DIR),
    package_dir={"": EXP_DIR},
    include_package_data=True,
    python_requires="==3.7.*",
    install_requires=[
        "essentia>=2.1b5;platform_system=='Linux'",  # audio signal processing
    ],
    extras_require={
        "development": [
            "black",
            "flake8",
            "pylint",
            "pylint-fail-under",
            "pytest",
            "rope",
            "tox"
        ],
        "demo": {
            "jupyter"
        }
    }
)
