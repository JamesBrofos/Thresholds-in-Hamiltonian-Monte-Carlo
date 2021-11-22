from setuptools import setup


# Import __version__ from code base.
exec(open("hmc/version.py").read())

setup(
    name="hmc",
    version=__version__,
    description="An analysis of convergence thresholds used in Riemannian manifold Hamiltonian Monte Carlo",
    author="James A. Brofos",
    author_email="james@brofos.org",
    url="http://brofos.org",
    keywords="generalized leapfrog riemannian manifold hamiltonian monte carlo thresholds fixed point newton",
)
