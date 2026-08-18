"""Convert standardized HERTA edge tables into a three-node PyG graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from herta.data.edge_tables import (
    MESSAGE_RELATIONS,
    NODE_TYPES,
    RELATIONS,
    EdgeTableBundle,
    validate_edge_table,
)


GRAPH_SCHEMA_VERSION = "herta_three_node_v1"

EDGE_FEATURE_COLUMNS = {
    "tf_peak": ("motif_score", "overlap_bp", "pvalue"),
    "peak_gene": ("distance", "prior_score"),
    "tf_target": ("prior_score",),
}

SPLIT_CODES = {"message": 0, "train": 1, "validation": 2, "test": 3, "query": 4}
LABEL_CODES = {
    "negative": 0.0,
    "observed": 1.0,
    "candidate": -1.0,
    "weak_positive": 1.0,
    "gold_positive": 1.0,
    "unknown": -1.0,
}

MESSAGE_EDGE_TYPES = {
    "cell_gene": RELATIONS["cell_gene"],
    "cell_peak": RELATIONS["cell_peak"],
}

QUERY_EDGE_TYPES = {
    "tp": RELATIONS["tf_peak"],
    "pg": RELATIONS["peak_gene"],
    "tg": RELATIONS["tf_target"],
}

REVERSE_RELATION_NAMES = {
    "expresses": "rev_expresses",
    "accessible": "rev_accessible",
    "binds": "rev_binds",
    "regulates": "rev_regulates",
}


@dataclass(frozen=True)
class QueryEdgeTensors:
    """Decoder-ready tensors corresponding row-for-row to one query table."""

    edge_type: tuple[str, str, str]
    edge_index: torch.Tensor
    prior_score: torch.Tensor
    distance: torch.Tensor
    table_row: torch.Tensor
    edge_features: torch.Tensor
    edge_feature_names: tuple[str, ...]
    split_code: torch.Tensor
    label: torch.Tensor


@dataclass
class HeterogeneousGraphBuildResult:
    """Three-node message graph plus regulatory decoder queries."""

    data: HeteroData
    features: dict[str, torch.Tensor]
    names: dict[str, list[str]]
    tf_gene_indices: torch.Tensor
    is_tf: torch.Tensor
    query_tensors: dict[str, QueryEdgeTensors]
    edge_tables: dict[str, pd.DataFrame]
    audit: dict[str, object]


def _node_ids(
    node_ids: Mapping[str, Sequence[str] | pd.Index],
) -> dict[str, list[str]]:
    missing = set(NODE_TYPES).difference(node_ids)
    extra = set(node_ids).difference(NODE_TYPES)
    if missing or extra:
        raise ValueError(
            f"node_ids must contain exactly {NODE_TYPES}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result: dict[str, list[str]] = {}
    for node_type in NODE_TYPES:
        values = pd.Index(node_ids[node_type], dtype="object").astype(str)
        if values.empty:
            raise ValueError(f"node_ids['{node_type}'] must not be empty.")
        if values.has_duplicates:
            raise ValueError(f"node_ids['{node_type}'] contains duplicate identifiers.")
        if (values.str.strip() == "").any():
            raise ValueError(f"node_ids['{node_type}'] contains empty identifiers.")
        result[node_type] = values.tolist()
    return result


def _features(
    node_features: Mapping[str, object],
    names: Mapping[str, list[str]],
) -> dict[str, torch.Tensor]:
    missing = set(NODE_TYPES).difference(node_features)
    extra = set(node_features).difference(NODE_TYPES)
    if missing or extra:
        raise ValueError(
            f"node_features must contain exactly {NODE_TYPES}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result: dict[str, torch.Tensor] = {}
    for node_type in NODE_TYPES:
        value = node_features[node_type]
        tensor = (
            value.detach().to(dtype=torch.float32, device="cpu")
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(np.asarray(value), dtype=torch.float32)
        )
        if tensor.ndim != 2 or tensor.shape[0] != len(names[node_type]):
            raise ValueError(
                f"{node_type} features must have shape "
                f"({len(names[node_type])}, n_features), got {tuple(tensor.shape)}."
            )
        if tensor.shape[1] < 1:
            raise ValueError(f"{node_type} features require at least one column.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{node_type} features must contain finite values.")
        result[node_type] = tensor.contiguous()
    return result


def _edge_index(
    table: pd.DataFrame,
    edge_type: tuple[str, str, str],
    indices: Mapping[str, Mapping[str, int]],
) -> torch.Tensor:
    source_type, _, target_type = edge_type
    source_index = indices[source_type]
    target_index = indices[target_type]
    missing_source = sorted(set(table["source_id"].astype(str)).difference(source_index))
    missing_target = sorted(set(table["target_id"].astype(str)).difference(target_index))
    if missing_source or missing_target:
        raise ValueError(
            f"Edge endpoints are absent from node_ids for {edge_type}: "
            f"source examples={missing_source[:3]}, target examples={missing_target[:3]}"
        )
    if table.empty:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(
        np.vstack(
            [
                table["source_id"]
                .astype(str)
                .map(source_index)
                .to_numpy(dtype=np.int64),
                table["target_id"]
                .astype(str)
                .map(target_index)
                .to_numpy(dtype=np.int64),
            ]
        )
    )


def _numeric_edge_attribute(
    table: pd.DataFrame,
    column: str,
    *,
    default: float = np.nan,
) -> torch.Tensor:
    if column not in table:
        values = np.full(len(table), default, dtype=np.float32)
    else:
        values = (
            pd.to_numeric(table[column], errors="coerce")
            .fillna(default)
            .to_numpy(dtype=np.float32)
        )
    return torch.from_numpy(values)


def _coded_edge_attribute(
    table: pd.DataFrame,
    column: str,
    mapping: Mapping[str, float | int],
    *,
    default: float,
) -> torch.Tensor:
    values = table[column].astype(str).map(mapping).fillna(default).to_numpy()
    dtype = torch.long if all(isinstance(value, int) for value in mapping.values()) else torch.float32
    return torch.as_tensor(values, dtype=dtype)


def _edge_feature_matrix(
    table: pd.DataFrame,
    relation_key: str,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    names = EDGE_FEATURE_COLUMNS.get(relation_key, ("prior_score",))
    columns = [
        _numeric_edge_attribute(table, name, default=0.0)
        for name in names
    ]
    matrix = (
        torch.stack(columns, dim=1)
        if columns
        else torch.empty((len(table), 0), dtype=torch.float32)
    )
    return matrix, tuple(names)


def _add_message_relation(
    data: HeteroData,
    table: pd.DataFrame,
    edge_type: tuple[str, str, str],
    indices: Mapping[str, Mapping[str, int]],
    relation_key: str,
) -> None:
    edge_index = _edge_index(table, edge_type, indices)
    weight_column = "edge_weight" if "edge_weight" in table else "prior_score"
    weights = _numeric_edge_attribute(table, weight_column, default=0.0)
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_weight = weights
    edge_features, feature_names = _edge_feature_matrix(table, relation_key)
    data[edge_type].edge_features = edge_features
    data[edge_type].edge_feature_names = feature_names
    table_rows = (
        pd.to_numeric(table["_table_row"], errors="raise").to_numpy(dtype=np.int64)
        if "_table_row" in table
        else table.index.to_numpy(dtype=np.int64)
    )
    data[edge_type].table_row = torch.as_tensor(table_rows, dtype=torch.long)
    data[edge_type].split_code = _coded_edge_attribute(
        table, "split", SPLIT_CODES, default=-1
    )
    data[edge_type].label = _coded_edge_attribute(
        table, "label_status", LABEL_CODES, default=-1.0
    )
    if "raw_value" in table:
        data[edge_type].raw_value = _numeric_edge_attribute(
            table, "raw_value", default=0.0
        )
    if "normalized_value" in table:
        data[edge_type].normalized_value = _numeric_edge_attribute(
            table, "normalized_value", default=0.0
        )
    reverse_name = REVERSE_RELATION_NAMES[edge_type[1]]
    reverse = (edge_type[2], reverse_name, edge_type[0])
    data[reverse].edge_index = edge_index.flip(0)
    data[reverse].edge_weight = weights.clone()
    data[reverse].edge_features = edge_features.clone()
    data[reverse].edge_feature_names = feature_names
    data[reverse].table_row = data[edge_type].table_row.clone()
    data[reverse].split_code = data[edge_type].split_code.clone()
    data[reverse].label = data[edge_type].label.clone()
    if "raw_value" in table:
        data[reverse].raw_value = data[edge_type].raw_value.clone()
    if "normalized_value" in table:
        data[reverse].normalized_value = data[edge_type].normalized_value.clone()


def _query_tensors(
    table: pd.DataFrame,
    edge_type: tuple[str, str, str],
    indices: Mapping[str, Mapping[str, int]],
    relation_key: str,
) -> QueryEdgeTensors:
    edge_features, feature_names = _edge_feature_matrix(table, relation_key)
    return QueryEdgeTensors(
        edge_type=edge_type,
        edge_index=_edge_index(table, edge_type, indices),
        prior_score=_numeric_edge_attribute(table, "prior_score", default=0.0),
        distance=_numeric_edge_attribute(table, "distance"),
        table_row=torch.arange(len(table), dtype=torch.long),
        edge_features=edge_features,
        edge_feature_names=feature_names,
        split_code=_coded_edge_attribute(table, "split", SPLIT_CODES, default=-1),
        label=_coded_edge_attribute(
            table, "label_status", LABEL_CODES, default=-1.0
        ),
    )


def _tf_mask(
    edge_tables: Mapping[str, pd.DataFrame],
    gene_names: list[str],
    tf_ids: Sequence[str] | pd.Index | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inferred: set[str] = set(edge_tables["tf_peak"]["source_id"].astype(str))
    if "tf_target" in edge_tables:
        inferred.update(edge_tables["tf_target"]["source_id"].astype(str))
    if tf_ids is None:
        selected = inferred
    else:
        values = pd.Index(tf_ids, dtype="object").astype(str)
        if values.has_duplicates:
            raise ValueError("tf_ids contains duplicate identifiers.")
        selected = set(values)
        missing_candidates = inferred.difference(selected)
        if missing_candidates:
            raise ValueError(
                "TF candidate sources are not marked in tf_ids: "
                f"{sorted(missing_candidates)[:5]}"
            )
    gene_index = {name: index for index, name in enumerate(gene_names)}
    missing_genes = selected.difference(gene_index)
    if missing_genes:
        raise ValueError(f"tf_ids are absent from gene nodes: {sorted(missing_genes)[:5]}")
    tf_gene_indices = torch.tensor(
        [gene_index[name] for name in gene_names if name in selected],
        dtype=torch.long,
    )
    is_tf = torch.zeros(len(gene_names), dtype=torch.bool)
    is_tf[tf_gene_indices] = True
    return tf_gene_indices, is_tf


def _isolated_node_counts(
    names: Mapping[str, list[str]],
    cell_gene: pd.DataFrame,
    cell_peak: pd.DataFrame,
) -> dict[str, int]:
    connected = {
        "cell": set(cell_gene["source_id"].astype(str))
        | set(cell_peak["source_id"].astype(str)),
        "gene": set(cell_gene["target_id"].astype(str)),
        "peak": set(cell_peak["target_id"].astype(str)),
    }
    return {
        node_type: len(set(node_names).difference(connected[node_type]))
        for node_type, node_names in names.items()
    }


def edge_tables_to_heterodata(
    edge_tables: EdgeTableBundle,
    *,
    node_ids: Mapping[str, Sequence[str] | pd.Index],
    node_features: Mapping[str, object],
    tf_ids: Sequence[str] | pd.Index | None = None,
    prior_message: bool = False,
) -> HeterogeneousGraphBuildResult:
    """Build a model-ready three-node HERTA ``HeteroData`` object.

    By default only observed cell-gene and cell-peak relations enter message
    passing. TF-peak, peak-gene, and optional TF-target edges are returned as
    decoder query tensors. ``prior_message=True`` is an explicit ablation that
    adds only train-split weak/gold TP or PG evidence.
    """

    if not isinstance(edge_tables, EdgeTableBundle):
        raise TypeError("edge_tables must be an EdgeTableBundle.")
    names = _node_ids(node_ids)
    features = _features(node_features, names)
    tables = {name: table.copy() for name, table in edge_tables.as_dict().items()}
    for relation_key, table in tables.items():
        validate_edge_table(table, expected_relation=relation_key)

    indices = {
        node_type: {name: index for index, name in enumerate(node_names)}
        for node_type, node_names in names.items()
    }
    data = HeteroData()
    data.schema_version = GRAPH_SCHEMA_VERSION
    for node_type in NODE_TYPES:
        data[node_type].num_nodes = len(names[node_type])
        data[node_type].x = features[node_type]

    for relation_key in MESSAGE_RELATIONS:
        _add_message_relation(
            data,
            tables[relation_key],
            MESSAGE_EDGE_TYPES[relation_key],
            indices,
            relation_key,
        )

    query_tensors = {
        "tp": _query_tensors(
            tables["tf_peak"], QUERY_EDGE_TYPES["tp"], indices, "tf_peak"
        ),
        "pg": _query_tensors(
            tables["peak_gene"], QUERY_EDGE_TYPES["pg"], indices, "peak_gene"
        ),
    }
    if "tf_target" in tables:
        query_tensors["tg"] = _query_tensors(
            tables["tf_target"], QUERY_EDGE_TYPES["tg"], indices, "tf_target"
        )

    tf_gene_indices, is_tf = _tf_mask(tables, names["gene"], tf_ids)
    data["gene"].is_tf = is_tf

    prior_message_counts = {"tp": 0, "pg": 0}
    if prior_message:
        for query_key, table_key in (("tp", "tf_peak"), ("pg", "peak_gene")):
            table = tables[table_key]
            accepted_labels = (
                {"weak_positive", "gold_positive"}
                if query_key == "tp"
                else {"candidate", "weak_positive", "gold_positive"}
            )
            include = table["split"].astype(str).eq("train") & table[
                "label_status"
            ].astype(str).isin(accepted_labels)
            train_prior = table.loc[include].copy()
            train_prior["_table_row"] = np.flatnonzero(include.to_numpy())
            train_prior = train_prior.reset_index(drop=True)
            prior_message_counts[query_key] = len(train_prior)
            _add_message_relation(
                data,
                train_prior,
                QUERY_EDGE_TYPES[query_key],
                indices,
                table_key,
            )

    data.validate(raise_on_error=True)
    audit = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "node_counts": {node_type: len(names[node_type]) for node_type in NODE_TYPES},
        "message_edge_counts": {
            relation_key: int(len(tables[relation_key]))
            for relation_key in MESSAGE_RELATIONS
        },
        "query_edge_counts": {
            query_key: int(query.edge_index.shape[1])
            for query_key, query in query_tensors.items()
        },
        "tf_count": int(is_tf.sum()),
        "isolated_message_nodes": _isolated_node_counts(
            names, tables["cell_gene"], tables["cell_peak"]
        ),
        "prior_message": bool(prior_message),
        "prior_message_edge_counts": prior_message_counts,
    }
    return HeterogeneousGraphBuildResult(
        data=data,
        features=features,
        names=names,
        tf_gene_indices=tf_gene_indices,
        is_tf=is_tf,
        query_tensors=query_tensors,
        edge_tables=tables,
        audit=audit,
    )


def build_heterogeneous_graph(
    edge_tables: EdgeTableBundle,
    *,
    node_ids: Mapping[str, Sequence[str] | pd.Index],
    node_features: Mapping[str, object],
    tf_ids: Sequence[str] | pd.Index | None = None,
    prior_message: bool = False,
) -> HeterogeneousGraphBuildResult:
    """Named graph-builder wrapper around :func:`edge_tables_to_heterodata`."""

    return edge_tables_to_heterodata(
        edge_tables,
        node_ids=node_ids,
        node_features=node_features,
        tf_ids=tf_ids,
        prior_message=prior_message,
    )
