# Import model implementations to populate mosa.models.registry via their
# @register_model decorators. Neither import triggers optional deps
# (mofapy2/mofax): mofa.model only imports them lazily inside methods.
from mosa.models.mofa.model import MOFAModel
from mosa.models.mosa.model import MOSAModel

__all__ = ["MOFAModel", "MOSAModel"]
