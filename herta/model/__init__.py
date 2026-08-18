"""Model components for HERTA-ctGRN."""

from herta.model.decoders import (
    GRNDecoderSuite,
    PeakGeneDecoder,
    TFGeneAggregation,
    TFGeneDecoder,
    TFPeakDecoder,
    TFPeakGeneAggregator,
)
from herta.model.grn_model import GRNModel, GRNModelOutput
from herta.model.hgt import HGTEncoder, HGTEncoderOutput
from herta.model.losses import (
    LossWeights,
    MultiTaskLossManager,
    MultiTaskLossOutput,
    Stage1LossConfig,
    Stage2LossConfig,
    contrastive_info_nce_loss,
    link_prediction_bce_loss,
    prototype_clustering_loss,
    tf_peak_gene_consistency_loss,
)

__all__ = [
    "HGTEncoder",
    "HGTEncoderOutput",
    "GRNDecoderSuite",
    "GRNModel",
    "GRNModelOutput",
    "LossWeights",
    "MultiTaskLossManager",
    "MultiTaskLossOutput",
    "Stage1LossConfig",
    "Stage2LossConfig",
    "PeakGeneDecoder",
    "TFGeneAggregation",
    "TFGeneDecoder",
    "TFPeakDecoder",
    "TFPeakGeneAggregator",
    "contrastive_info_nce_loss",
    "link_prediction_bce_loss",
    "prototype_clustering_loss",
    "tf_peak_gene_consistency_loss",
]
from herta.model.cell_conditioned_decoders import (
    CellConditionedPeakGeneDecoder,
    CellConditionedTFPeakDecoder,
    Stage2CellConditionedModel,
)

__all__ += [
    "CellConditionedPeakGeneDecoder",
    "CellConditionedTFPeakDecoder",
    "Stage2CellConditionedModel",
]
