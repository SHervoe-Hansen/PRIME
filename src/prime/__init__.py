"""PRIME: Protonation-number Reweighting in Multi-pH Ensembles.

Free energies, titration curves and protonation-number distributions at any pH
from the protonation numbers recorded in constant-pH simulations, with the
diagnostics and the reference thermodynamic-integration estimators of

    S. Hervø-Hansen and N. Matubayasi,
    "Protonation-number Reweighting for pH-Dependent Free Energy Calculations".

See ``help(prime.core)`` for the overview of the functions. The tests in
``tests/`` check the package against an exactly solvable model of independent sites.
"""
from .core import *                                        # noqa: F401,F403
from .core import __all__, __version__                     # noqa: F401
