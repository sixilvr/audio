from distutils.core import setup

setup(name = "audio",
    version = "1.5",
    description = "A Python module for interacting with audio data and creating music",
    author = "SL",
    url = "https://github.com/sixilvr/audio",
    packages = ["audio"],
    install_requires = ["numpy", "scipy", "matplotlib"],
    provides = ["beatgen"]
)
