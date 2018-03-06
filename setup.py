from distutils.core import setup, Extension

extensions = [
    Extension(
        "prob",
        ["mclearn/prob/bind.c", "mclearn/prob/prob.c"],
        extra_compile_args = ["-fpermissive"]
    )
]

setup(
    name='MCLearn',
    version='0.1.0',
    author='A. Krog',
    author_email='akrog@cs.aau.dk',
    packages=['mclearn'],
    description='Learning Marcov Chains using genetic programming',
    ext_modules = extensions
)