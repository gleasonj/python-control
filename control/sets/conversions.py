from .hyperrectangle import Hyperrectangle

from .supporsets import SupportSet

from .polytopes import HPolytope, VPolytope

def convert(A, B):
    if isinstance(A, Hyperrectangle):
        if B == A:
            pass
