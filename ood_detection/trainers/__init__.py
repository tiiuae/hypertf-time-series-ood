from .angular_auxiliary_contrastive_trainer import AngularAuxiliaryContrastiveTrainer
from .base_trainer import BaseTrainer
from .contrastive_trainer import ContrastiveTrainer

IMPLEMENTED_TRAINERS = {
    "BaseTrainer": BaseTrainer,
    "AngularAuxiliaryContrastiveTrainer": AngularAuxiliaryContrastiveTrainer,
    "ContrastiveTrainer": ContrastiveTrainer,
}


def init_trainer(trainer_name: str, **kwargs):
    """Initialize the trainer."""
    try:
        trainer = IMPLEMENTED_TRAINERS[trainer_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{trainer_name} is not implemented. " + f"Available Trainers: {IMPLEMENTED_TRAINERS.keys()}"
        ) from exc
    return trainer
