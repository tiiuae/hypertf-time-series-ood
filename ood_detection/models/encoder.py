from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tsai.all import InceptionTime, InceptionTimePlus, ResCNN, ResNet

from ood_detection.models.pooling import get_pooling
from ood_detection.models.ts2vec_encoder import generate_binomial_mask
from ood_detection.utils.model_ops import init_norm


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device
    encoder: nn.Module
    fft_encoder: nn.Module | None = None
    feature_dim: int
    num_classes: int | None = None
    global_pooling: str = "max"  # "max", "avg", "ppv", "max_avg", "max_ppv"
    dropout_rate: float = 0.0
    input_channels: int | None = None
    input_proj_dim: int | None = None
    input_proj_mask_proba: float = 1.0
    input_proj_norm_type: str | None = None  # 'layer', None
    output_proj_head_enabled: bool = False
    output_proj_bias: bool = True
    output_proj_num_layers: int = 1
    output_proj_hidden_dim: int = 128
    output_proj_output_dim: int = 128
    output_proj_norm_type: str | None = None  # 'batch', 'group', 'layer', None
    ce_head: bool = False
    fft_on_raw: bool = False


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3, norm_type="batch", bias=True, num_groups=4):
        """
        Projection Head with configurable depth and normalization.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
            depth (int): Number of layers in the projection head.
            norm_type (str): Normalization type: 'batch', 'group', 'layer', or 'none'.
            bias (bool): Whether to include bias in linear layers.
            num_groups (int): Number of groups for GroupNorm (if used).
        """
        super().__init__()
        layers = []

        if depth == 1:
            layers.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            for i in range(depth - 1):
                in_dim = input_dim if i == 0 else hidden_dim
                layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))

                if norm_type is not None:
                    layers.append(init_norm(norm_type, hidden_dim, num_groups))

                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FFTEncoder(nn.Module):
    def __init__(self, fft_encoder, use_phase=True, eps=1e-6):
        super().__init__()
        self.fft_encoder = fft_encoder
        self.use_phase = use_phase
        self.eps = eps

    def forward(self, x):
        # x: [B, C, S]
        fft_coeffs = torch.fft.rfft(x, dim=-1)  # [B, C, F]

        magnitude = torch.abs(fft_coeffs)
        log_magnitude = torch.log1p(magnitude + self.eps)

        if self.use_phase:
            phase_cos = fft_coeffs.real / (magnitude + self.eps)
            phase_sin = fft_coeffs.imag / (magnitude + self.eps)
            features = torch.cat([log_magnitude, phase_cos, phase_sin], dim=1)  # [B, 3C, F]
        else:
            features = log_magnitude  # [B, C, F]

        return self.fft_encoder(features)


class Classifier(nn.Module):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = cfg.encoder
        self.fft_encoder = cfg.fft_encoder
        self.global_pooling = cfg.global_pooling
        self.dropout = nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate > 0.0 else None
        self.output_proj_output_dim = cfg.output_proj_output_dim
        self.input_proj_mask_proba = cfg.input_proj_mask_proba

        # Projection layer (only if input_proj_dim is specified)
        self.input_fc = None
        if cfg.input_proj_dim is not None:
            if cfg.input_channels is None:
                raise ValueError("`input_channels` must be specified when `input_proj_dim` is provided.")

            self.input_fc = nn.Sequential(
                nn.Linear(cfg.input_channels, cfg.input_proj_dim, bias=True),
                nn.LayerNorm(cfg.input_proj_dim) if cfg.input_proj_norm_type == "layer" else nn.Identity(),
            )
            self.input_proj_dim = cfg.input_proj_dim

        # concat_pool layers
        self.pooling = get_pooling(cfg.global_pooling, cfg.feature_dim)

        # Calculate the actual output dimension after pooling
        self.pooled_feature_dim = cfg.feature_dim * self.pooling.output_multiplier

        # Define the input projection for the output projection head
        if cfg.output_proj_head_enabled:
            self.output_proj_head = ProjectionHead(
                input_dim=self.pooled_feature_dim,
                hidden_dim=cfg.output_proj_hidden_dim,
                output_dim=cfg.output_proj_output_dim,
                depth=cfg.output_proj_num_layers,
                norm_type=cfg.output_proj_norm_type,
                bias=cfg.output_proj_bias,
            )
        else:
            self.output_proj_head = nn.Identity()

        dim = self.output_proj_output_dim if cfg.output_proj_head_enabled else self.pooled_feature_dim
        self.classifier = self._get_classifier(dim, cfg.num_classes)

        # FFT projection head (optional)
        if self.fft_encoder is not None:
            if cfg.output_proj_head_enabled:
                self.fft_output_proj_head = ProjectionHead(
                    input_dim=self.pooled_feature_dim,
                    hidden_dim=cfg.output_proj_hidden_dim,
                    output_dim=cfg.output_proj_output_dim,
                    depth=cfg.output_proj_num_layers,
                    norm_type=cfg.output_proj_norm_type,
                    bias=cfg.output_proj_bias,
                )
            else:
                self.fft_output_proj_head = nn.Identity()
        else:
            self.fft_output_proj_head = None

    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape [Batch, Channels, Seq_len]/ [B, C, S]
        """
        x = x.transpose(1, 2)  # to [B, S, C]
        nan_mask = ~x.isnan().any(dim=-1)  # [B, S], bool

        # Expand mask to 3D explicitly so ONNX shape inference works
        mask3 = nan_mask.unsqueeze(-1).expand_as(x)  # [B, S, C]
        # Use torch.where with same-shaped tensors
        x = torch.where(mask3, x, torch.zeros_like(x))

        # Save cleaned raw for the FFT branch
        x_raw = x.transpose(1, 2)  # [B, C, S]

        # Apply input projection (if defined)
        if self.input_fc is not None:
            # Expand on the feature dimension
            x = self.input_fc(x)  # Project input: [batch, seq_len, input_proj_dim]

            # Transpose back after projection
            x = x.transpose(1, 2)  # Back to [batch, input_proj_dim, seq_len]

            # Generate binomial mask (only in training)
            if self.training:
                mask = generate_binomial_mask(x.size(0), x.size(2), p=self.input_proj_mask_proba).to(x.device)
            else:
                mask = torch.ones_like(nan_mask)

            # No in-place ops; ONNX maps to LogicalAnd
            mask = torch.logical_and(mask, nan_mask)  # [B, S]
            mask = mask.unsqueeze(1).expand(x.size(0), 1, x.size(2))  # [B,1,S]
            x = torch.where(mask, x, torch.zeros_like(x))  # [B, D_proj, S]
        else:
            # Transpose back to the original format
            x = x.transpose(1, 2)  # Back to [batch, feat_dim, seq_len]

        # Compute features from time domain
        temporal_features = self.encoder(x)

        # Compute features from frequency domain
        if self.fft_encoder is not None:
            if self.cfg.fft_on_raw:
                temporal_fft_features = self.fft_encoder(x_raw)
            else:
                temporal_fft_features = self.fft_encoder(x)
        else:
            temporal_fft_features = None

        # Apply pooling to raw features
        instance_features = self.pooling(temporal_features) if temporal_features.ndim == 3 else temporal_features

        # Apply pooling to FFT features
        fft_instance_features = (
            self.pooling(temporal_fft_features)
            if temporal_fft_features is not None and temporal_fft_features.ndim == 3
            else temporal_fft_features
        )

        # Dropout
        if self.dropout:
            instance_features = self.dropout(instance_features)
            # fft_instance_features = self.dropout(fft_instance_features)  # adding dropout to fft hurts

        # Raw projection
        if self.output_proj_head:
            projected_features = self.output_proj_head(instance_features)
        else:
            projected_features = None

        # FFT projection
        if fft_instance_features is not None:
            fft_projected_features = (
                self.fft_output_proj_head(fft_instance_features) if self.fft_output_proj_head is not None else None
            )
        else:
            fft_projected_features = None

        # Concatenate raw + fft pooled features
        if fft_instance_features is not None:
            instance_features = torch.cat(
                [instance_features, fft_instance_features],
                dim=0,  # NOTE at batch dim
            )

        # Concatenate raw + fft projected features
        if projected_features is not None and fft_projected_features is not None:
            projected_features = torch.cat(
                [projected_features, fft_projected_features],
                dim=0,  # NOTE at batch dim
            )

        # Compute logits on the final projected features
        if projected_features is not None:
            logits = self._compute_logits(projected_features)
        else:
            logits = self._compute_logits(instance_features)

        return temporal_features, instance_features, projected_features, logits

    def _compute_logits(self, features: torch.Tensor):
        """
        features can either be instance_features or projected_features
        """
        logits = self.classifier(features)
        return logits

    def _get_classifier(self, feature_dim: int, num_classes: int):
        classifier = nn.Linear(feature_dim, num_classes)
        return classifier


class AngularClassifier(Classifier):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__(cfg)
        dim = self.output_proj_output_dim if cfg.output_proj_head_enabled else self.pooled_feature_dim
        self.classifier = self._get_classifier(dim, cfg.num_classes)

    def _get_classifier(self, feature_dim: int, num_classes: int):
        classifier = nn.Parameter(torch.randn(num_classes, feature_dim))  # class prototypes
        nn.init.xavier_uniform_(classifier)
        return classifier

    def _compute_logits(self, features: torch.Tensor):
        """
        If output_proj_head_enabled then features are projected_features else instance_features
        """
        # Normalize feature vectors and class weights
        features = F.normalize(features, p=2, dim=1)
        classifier_norm = F.normalize(self.classifier, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.matmul(features, classifier_norm.T)

        # Clamp logits to prevent numerical instability
        logits = logits.clamp(-1.0, 1.0)

        return logits


class CIDERAngularClassifier(AngularClassifier):
    def _get_classifier(self, feature_dim: int, num_classes: int):
        # class prototypes when using cider loss are registered buffers instead of parameters
        classifier = torch.randn(num_classes, feature_dim).to(self.cfg.device)
        return classifier


class ContrastiveEuclideanClassifier(Classifier):
    def __init__(self, cfg: ClassifierConfig):
        """
        Contrastive Classifier:
        - Does NOT have a classification layer.
        - Returns **projected features** instead of logits.
        """
        super().__init__(cfg)
        # since we use instance_features for CE, not projected_features, the dim must be self.pooled_feature_dim
        dim = self.pooled_feature_dim
        self.classifier = self._get_classifier(dim, cfg.num_classes)

    def _compute_logits(self, proj_inst_features: torch.Tensor):
        # Check if shape matches expected classifier input
        if proj_inst_features.shape[1] != self.classifier.in_features:
            return None
        return self.classifier(proj_inst_features)

    def forward(self, x: torch.Tensor):
        # Reuse base Classifier forward pass to get encoder + supcon projection
        temporal_features, instance_features, proj_supcon, _ = super().forward(x)

        # Use a separate projection head for CE
        logits = self._compute_logits(instance_features)

        return temporal_features, instance_features, proj_supcon, _, logits


class ContrastiveHyperClassifier(Classifier):
    def __init__(self, cfg: ClassifierConfig):
        """
        Contrastive Classifier with:
        - existing output_proj_head used for SupCon
        - new ce_proj_head used for classification
        - learnable, normalized prototypes
        """
        super().__init__(cfg)

        dim = self.output_proj_output_dim if cfg.output_proj_head_enabled else self.pooled_feature_dim

        # Projection head for CE classification (separate from SupCon head)
        if self.cfg.ce_head:
            self.ce_proj_head = ProjectionHead(
                input_dim=self.pooled_feature_dim,
                hidden_dim=cfg.output_proj_hidden_dim,
                output_dim=cfg.output_proj_output_dim,
                depth=cfg.output_proj_num_layers,
                norm_type=cfg.output_proj_norm_type,
                bias=cfg.output_proj_bias,
            )

        # Learnable class prototypes
        self.classifier = self._get_classifier(dim, cfg.num_classes)

    def _get_classifier(self, feature_dim: int, num_classes: int):
        classifier = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(classifier)
        return classifier

    def _compute_logits(self, projected_features: torch.Tensor):
        # Normalize both features and prototypes
        projected_features = F.normalize(projected_features, p=2, dim=1)
        classifier_norm = F.normalize(self.classifier, p=2, dim=1)

        # Cosine similarity logits
        logits = torch.matmul(projected_features, classifier_norm.T)
        return logits.clamp(-1.0, 1.0)

    def forward(self, x: torch.Tensor):
        # Reuse base Classifier forward pass to get encoder + supcon projection
        temporal_features, instance_features, proj_supcon, logits = super().forward(x)

        proj_ce = None
        if self.cfg.ce_head:
            # Use a separate projection head for CE
            proj_ce = self.ce_proj_head(instance_features)
            logits = self._compute_logits(proj_ce)  # overwrite logits

        return temporal_features, instance_features, proj_supcon, proj_ce, logits


class AngularAuxiliarySingleViewClassifier(AngularClassifier):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__(cfg)

        # Additional projection head for binary inlier/outlier separation
        self.outlier_proj_head = ProjectionHead(
            input_dim=self.pooled_feature_dim,
            hidden_dim=cfg.output_proj_hidden_dim,
            output_dim=cfg.output_proj_output_dim,
            depth=cfg.output_proj_num_layers,
            norm_type=cfg.output_proj_norm_type,
            bias=cfg.output_proj_bias,
        )

        # single direction vector; two prototypes are +v and -v by construction
        proto = F.normalize(torch.randn(cfg.feature_dim), dim=0)
        # store one param; derive the pair in forward() so they stay opposite
        self._proto_dir = nn.Parameter(proto)

        # scale for the OE logits (unbounded “logit-ification” of cosine)
        self.oe_scale = nn.Parameter(torch.tensor(10.0))  # init in [8, 32] territory works well

    def forward(self, x: torch.Tensor):
        # Get base features and projections from parent forward
        # logits may not make sense for outliers
        temporal_features, instance_features, projected_features, logits = super().forward(x)

        # project to OE space
        oe_feats = self.outlier_proj_head(instance_features)
        oe_feats = F.normalize(oe_feats, p=2, dim=1)

        # strict opposite prototypes
        v = F.normalize(self._proto_dir, dim=0)
        in_out_protos = torch.stack([v, -v], dim=0)  # [2, D]

        # cosine -> scale -> logits (NO clamp)
        cos = torch.matmul(oe_feats, in_out_protos.T)  # [B,2] in [-1,1]
        in_out_logits = self.oe_scale.clamp_min(1e-3) * cos  # [B,2] unbounded-ish

        return temporal_features, instance_features, projected_features, logits, in_out_logits


class AngularAuxiliaryContrastiveClassifier(AngularClassifier):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__(cfg)

        # Additional projection head for binary inlier/outlier separation
        self.outlier_proj_head = ProjectionHead(
            input_dim=self.pooled_feature_dim,
            hidden_dim=cfg.output_proj_hidden_dim,
            output_dim=cfg.output_proj_output_dim,
            depth=cfg.output_proj_num_layers,
            norm_type=cfg.output_proj_norm_type,
            bias=cfg.output_proj_bias,
        )

    def forward(self, x: torch.Tensor):
        # Get base features and projections from parent forward
        temporal_features, instance_features, projected_features, logits = super().forward(x)

        # Pass it through secondary projection head for contrastive with auxiliary
        sec_projected_features = self.outlier_proj_head(instance_features)
        sec_projected_features = F.normalize(sec_projected_features, p=2, dim=1)

        return temporal_features, instance_features, projected_features, logits, sec_projected_features


def _parse_tst_hparams(name: str):
    """
    Allow names like:
      'tst'                      -> defaults
      'tst_d256_h8_l6_p16'       -> d_model=256, heads=8, depth=6, patch_len=16
      'vit1d_p32'                -> only patch_len=32, others default
      'tst_d128_h4_4_p1'         -> best
    """
    d_model, n_heads, depth, patch_len = 128, 4, 4, 1
    # simple tokens like d256 h8 l6 p16
    for tok in name.split("_"):
        if tok.startswith("d") and tok[1:].isdigit():
            d_model = int(tok[1:])
        elif tok.startswith("h") and tok[1:].isdigit():
            n_heads = int(tok[1:])
        elif tok.startswith("l") and tok[1:].isdigit():
            depth = int(tok[1:])
        elif tok.startswith("p") and tok[1:].isdigit():
            patch_len = int(tok[1:])
    return d_model, n_heads, depth, patch_len


def init_encoder(name: str, input_length: int, input_channels: int, verbose: bool = True):
    # Choose the encoder model
    name = name.lower()
    if name == "inceptiontime":
        model = InceptionTime(c_in=input_channels, nf=32, c_out=1)
    elif name == "inceptiontimeplus":
        model = InceptionTimePlus(c_in=input_channels, nf=32, c_out=1)
    elif name == "resnet":
        model = ResNet(c_in=input_channels, c_out=1)
    elif name == "rescnn":
        model = ResCNN(c_in=input_channels, c_out=1)
    else:
        raise ValueError(
            f"Unsupported encoder model: {name}. Available options: inceptiontime, inceptiontimeplus, inception_gru, singleblockinception, tst, resnet."
        )

    # Remove the classifier head (linear layer) robustly
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    if hasattr(model, "lin"):
        model.lin = torch.nn.Identity()
    if hasattr(model, "outlinear"):
        model.outlinear = torch.nn.Identity()
    if hasattr(model, "linear"):
        model.linear = torch.nn.Identity()

    # Raise an error if neither 'head' nor 'fc' exists
    if (
        not hasattr(model, "head")
        and not hasattr(model, "fc")
        and not hasattr(model, "lin")
        and not hasattr(model, "outlinear")
        and not hasattr(model, "linear")
    ):
        print(model)
        raise ValueError(
            f"Model {type(model).__name__} does not have a recognized classifier head ('head' or 'fc' or 'fc')."
        )

    # Compute feature dimension
    dummy_input = torch.randn(1, input_channels, input_length)  # (batch_size, channels, seq_len)

    output = model(dummy_input)
    model.feature_dim = output.shape[1] if output.ndim == 3 else output.shape[-1]

    if verbose:
        print(f"Initialized encoder (without classifier): {model.__class__.__name__}")
        print(f"Feature Dimension: {model.feature_dim}")

    return model


def init_model(config: DictConfig, name: str, dataset: Dataset, device: torch.device, verbose: bool = True):
    """
    Init model that is a classifier that wraps an encoder
    """
    input_length = dataset.ts_length
    input_channels = dataset.num_features

    # Read boolean flag for input projection from config
    dropout_rate = config.model.args.get("dropout_rate", 0.0)
    global_pooling = config.model.args.get("global_pooling", "max")
    input_projection_cfg = config.model.args.get("input_projection", {})
    project_input = input_projection_cfg.get("enabled", False)
    input_proj_dim = input_projection_cfg.get("dim", None)
    input_proj_mask_proba = 1.0 - float(input_projection_cfg.get("training_mask_probability", 0.0))
    input_proj_norm_type = input_projection_cfg.get("norm_type", None)
    output_proj_head_cfg = config.model.args.get("output_projection", {})
    output_proj_head_enabled = output_proj_head_cfg.get("enabled", False)
    output_proj_num_layers = output_proj_head_cfg.get("num_layers", 1)
    output_proj_bias = output_proj_head_cfg.get("bias", True)
    output_proj_norm_type = output_proj_head_cfg.get("norm_type", None)
    ce_head = config.model.args.get("ce_head", False)

    if project_input:
        if input_proj_dim is None:
            raise ValueError("Input projection is enabled but 'dim' is not specified in the config.")
        input_proj_dim = max(input_proj_dim, input_channels)
    else:
        input_proj_dim = None

    # Initialize the encoder for the raw signal
    encoder_input_dim = input_proj_dim if project_input else input_channels
    encoder = init_encoder(name, input_length, encoder_input_dim, verbose=verbose)

    # Special case: FFT encoder for frequency domain inputs
    fft_encoder = None
    if config.model.args.get("use_fft", False):
        if config.model.args.fft_on_raw:
            ts_encoder = init_encoder(name, input_length, input_channels, verbose=verbose)
        else:
            fft_encoder_dim = 3 * encoder_input_dim if config.model.args.use_phase else encoder_input_dim
            ts_encoder = init_encoder(name, input_length, fft_encoder_dim, verbose=verbose)
        fft_encoder = FFTEncoder(ts_encoder, use_phase=config.model.args.use_phase)

    # Set set default hidden and output dimensions for the projection head
    output_proj_hidden_dim = output_proj_head_cfg.get("hidden_dim", encoder.feature_dim)
    output_proj_output_dim = output_proj_head_cfg.get("output_dim", encoder.feature_dim)

    # Get trainer type from config
    trainer_type = config.trainer.type  # e.g., "BaseTrainer", "ContrastiveTrainer"

    # Determine the classifier based on the trainer type and cosine flag
    if trainer_type == "ContrastiveTrainer":
        # use ContrastiveEuclideanClassifier for SupCon (euclidean)
        clf_class = ContrastiveHyperClassifier if config.model.args.cosine else ContrastiveEuclideanClassifier
    elif trainer_type == "AngularAuxiliaryContrastiveTrainer":
        clf_class = AngularAuxiliaryContrastiveClassifier
    else:
        clf_class = Classifier

    cfg = ClassifierConfig(
        device=device,
        encoder=encoder,
        fft_encoder=fft_encoder,
        feature_dim=encoder.feature_dim,
        num_classes=dataset.num_classes,
        global_pooling=global_pooling,
        dropout_rate=dropout_rate,
        input_channels=input_channels,
        input_proj_dim=input_proj_dim,
        input_proj_mask_proba=input_proj_mask_proba,
        input_proj_norm_type=input_proj_norm_type,
        output_proj_head_enabled=output_proj_head_enabled,
        output_proj_bias=output_proj_bias,
        output_proj_num_layers=output_proj_num_layers,
        output_proj_hidden_dim=output_proj_hidden_dim,
        output_proj_output_dim=output_proj_output_dim,
        output_proj_norm_type=output_proj_norm_type,
        ce_head=ce_head,
        fft_on_raw=config.model.args.get("fft_on_raw", False),
    )
    model = clf_class(cfg)

    if verbose:
        # Print model size in millions of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Size: {total_params / 1e6:.2f}M parameters")

    return model
