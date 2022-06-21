from setuptools import setup, find_packages

setup(
    name='BPTT_TF',
    version='0.1.0',
    description='Training (PL)RNNs for dynamical systems reconstruction using BPTT & Teacher Forcing.',
    author='Jonas Mikhaeil, Florian Hess',
    author_email='Jonas.Mikhaeil@zi-mannheim.de, Florian.Hess@zi-mannheim.de',
    url = "https://gitlab.zi.local/Jonas.Mikhaeil/bptt",
    packages=find_packages()
)
