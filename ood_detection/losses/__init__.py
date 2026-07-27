from copy import deepcopy

from omegaconf.listconfig import ListConfig
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .aux_contrastive_loss import AuxiliaryContrastiveLoss
from .compactness_loss import CompactnessLoss
from .contrastive_loss import InfoNCELoss, NTXentLoss, SupConLoss
from .fft_consistency_loss import FFTConsistencyLoss
from .multiobjective_loss import MultiObjectiveLoss
from .scaled_ce_loss import RescaledTemperatureCELoss, TemperatureScaledCELoss
from .separation_loss import SeparationLoss


def init_specific_loss(
    loss_type: str,
    loss_args: dict,
    device: torch.device,
    num_classes: int = None,
    feat_dim: int = None,
    prototypes: torch.Tensor = None,
    class_weights: torch.Tensor = None,
):
    """
    Initialize a specific loss function based on the provided type and arguments.
    Pass runtime arguments separately, e.g., num_classes, feat_dim, prototypes, class_weights.
    """
    loss_args = loss_args or {}

    # cross entropy loss
    if loss_type == "CrossEntropyLoss":
        return CrossEntropyLoss(weight=class_weights, **loss_args)
    if loss_type == "TemperatureScaledCELoss":
        return TemperatureScaledCELoss(class_weights=class_weights, **loss_args)
    if loss_type == "RescaledTemperatureCELoss":
        return RescaledTemperatureCELoss(class_weights=class_weights, **loss_args)
    # compactness/separation losses
    if loss_type == "CompactnessLoss":
        return CompactnessLoss(num_classes=num_classes, **loss_args)
    if loss_type == "SeparationLoss":
        return SeparationLoss(num_classes=num_classes, prototypes=prototypes, **loss_args)
    # binary class based losses
    if loss_type == "BCEWithLogitsLoss":
        return BCEWithLogitsLoss(**loss_args)
    # contrastive losses
    if loss_type == "InfoNCELoss":
        return InfoNCELoss(**loss_args)
    if loss_type == "NTXentLoss":
        return NTXentLoss(**loss_args)
    if loss_type == "SupConLoss":
        return SupConLoss(class_weights=class_weights, **loss_args)
    if loss_type == "AuxiliaryContrastiveLoss":
        return AuxiliaryContrastiveLoss(**loss_args)
    if loss_type == "FFTConsistencyLoss":
        return FFTConsistencyLoss(**loss_args)
    raise NotImplementedError(f"Loss {loss_type} is not implemented.")


def _prepare_loss_args(loss_cfg, device, class_weights):
    """Prepare the arguments for the loss function based on the configuration.
    Drop any args here that are not needed for the specific loss function.

    Args:
        loss_cfg (dict): Configuration for the loss function.
        device (torch.device): Device to which the loss should be moved.
        class_weights (List[float]): Class weights if applicable.
    Returns:
        tuple: A tuple containing the loss arguments and class weights tensor.
    """
    loss_args = deepcopy(loss_cfg.get("args", {}))
    if loss_args.get("use_class_weights", False):
        if class_weights is None:
            raise ValueError(f"use_class_weights=True but no class_weights provided for {loss_cfg}.")
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    else:
        class_weights_tensor = None

    # drop all args that are not needed for the specific loss function
    loss_args.pop("use_class_weights", None)
    return loss_args, class_weights_tensor


def init_loss(loss_config, device, num_classes=None, feat_dim=None, prototypes=None, class_weights=None):
    """Initialize the loss function based on the configuration."""

    if loss_config.type == "MultiObjectiveLoss":
        losses = loss_config.args.losses
        lambdas = loss_config.args.lambdas
        lambda_warmups = loss_config.args.lambda_warmup_epochs

        # convert to lists if not already
        losses = losses if isinstance(losses, list | ListConfig) else [losses]
        lambdas = lambdas if isinstance(lambdas, list | ListConfig) else [lambdas]
        lambda_warmups = lambda_warmups if isinstance(lambda_warmups, list | ListConfig) else [lambda_warmups]
        if not (len(losses) == len(lambdas) == len(lambda_warmups)):
            raise ValueError("Each loss must have a corresponding lambda and warmup_epochs value.")

        loss_dict, lambda_dict, warmup_dict = {}, {}, {}
        for _, (loss, loss_lambda, loss_lambda_warmup) in enumerate(zip(losses, lambdas, lambda_warmups, strict=True)):
            loss_type = loss.type
            loss_args, class_weights_tensor = _prepare_loss_args(loss, device, class_weights)
            loss_instance = init_specific_loss(
                loss_type,
                loss_args,
                device=device,
                num_classes=num_classes,
                feat_dim=feat_dim,
                prototypes=prototypes,
                class_weights=class_weights_tensor,
            )
            loss_dict[loss_type] = loss_instance
            lambda_dict[loss_type] = loss_lambda
            warmup_dict[loss_type] = loss_lambda_warmup

        return MultiObjectiveLoss(loss_dict, lambda_dict, warmup_dict)
    else:
        loss_type = loss_config.type
        loss_args, class_weights = _prepare_loss_args(loss_config, device, class_weights)
        return init_specific_loss(
            loss_type,
            loss_args,
            device=device,
            num_classes=num_classes,
            feat_dim=feat_dim,
            prototypes=prototypes,
            class_weights=class_weights,
        )
