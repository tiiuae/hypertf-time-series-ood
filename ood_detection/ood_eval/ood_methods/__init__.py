from .embedding_center_score import EmbeddingCenterScore
from .energy_score import EnergyScore
from .gen_score import GENScore
from .maha_score import MahalanobisScore
from .max_logit_score import MaxLogitScore
from .msp_score import MSPScore
from .nn_score import NearestNeighborScore


def init_ood_eval_method(ood_eval_name: str, **ood_eval_kwargs: dict):
    """
    Initialize OOD eval method
    """
    if ood_eval_name == "MaxLogitScore":
        return MaxLogitScore()
    elif ood_eval_name == "MSPScore":
        return MSPScore()
    elif ood_eval_name == "EnergyScore":
        return EnergyScore(**ood_eval_kwargs)
    elif ood_eval_name == "GENScore":
        return GENScore(**ood_eval_kwargs)
    elif ood_eval_name == "NearestNeighborScore":
        return NearestNeighborScore(**ood_eval_kwargs)
    elif ood_eval_name == "EmbeddingCenterScore":
        return EmbeddingCenterScore(**ood_eval_kwargs)
    elif ood_eval_name == "MahalanobisScore":
        return MahalanobisScore(**ood_eval_kwargs)
    else:
        raise NotImplementedError(f"OOD evaluator {ood_eval_name} not implemented.")
