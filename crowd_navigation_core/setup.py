from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['my_tiago_controller'],
    package_dir={'': 'src'}
)
setup(**d)