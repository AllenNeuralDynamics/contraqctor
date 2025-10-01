from importlib.metadata import PackageNotFoundError, version

from . import contract as contract
from . import qc as qc

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
