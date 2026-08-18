"""Plotting helpers for HERTA workflow outputs."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from herta import settings


def _save_or_return(fig: plt.Figure, filename: str | None) -> plt.Axes:
    if settings.save_fig and filename is not None:
        settings.figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.figdir / filename, bbox_inches="tight", dpi=settings.figure_params.get("dpi", 150))
    return fig.axes[0]


def training_metrics(metrics: pd.DataFrame | str | Path, y: str = "loss") -> plt.Axes:
    """Plot training metrics from a DataFrame or ``train_metrics.csv`` path."""

    df = pd.read_csv(metrics) if isinstance(metrics, str | Path) else metrics
    fig, ax = plt.subplots(figsize=settings.figure_params.get("figsize", (6, 3.5)))
    sns.lineplot(data=df, x="epoch", y=y, marker="o", ax=ax)
    ax.set_title(f"Training {y}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(y)
    return _save_or_return(fig, f"training_{y}.png")


def grn_scores(edges: pd.DataFrame | str | Path) -> plt.Axes:
    """Plot the distribution of inferred GRN scores."""

    df = pd.read_parquet(edges) if isinstance(edges, str | Path) else edges
    fig, ax = plt.subplots(figsize=settings.figure_params.get("figsize", (6, 3.5)))
    sns.histplot(df["score"], bins=30, kde=True, ax=ax)
    ax.set_title("GRN score distribution")
    ax.set_xlabel("score")
    return _save_or_return(fig, "grn_scores.png")


def mediator_peaks(edges: pd.DataFrame | str | Path) -> plt.Axes:
    """Plot mediator peak counts for inferred TF-gene edges."""

    df = pd.read_parquet(edges) if isinstance(edges, str | Path) else edges
    if "n_mediator_peaks" not in df:
        raise ValueError("Expected a 'n_mediator_peaks' column in GRN edges.")
    fig, ax = plt.subplots(figsize=settings.figure_params.get("figsize", (6, 3.5)))
    sns.countplot(data=df, x="n_mediator_peaks", color="#4C78A8", ax=ax)
    ax.set_title("Mediator peak support")
    ax.set_xlabel("number of mediator peaks")
    ax.set_ylabel("TF-gene edges")
    return _save_or_return(fig, "mediator_peaks.png")


def build_tf_target_graph(
    edges: pd.DataFrame | str | Path,
    tf_col: str = "tf",
    target_col: str | None = None,
    score_col: str = "score",
    weight_col: str | None = None,
    sign_col: str | None = None,
    evidence_col: str | None = None,
    cell_type: str | None = None,
    selected_tfs: list[str] | set[str] | tuple[str, ...] | None = None,
    min_score: float | None = None,
    top_k_edges: int | None = None,
    top_k_targets_per_tf: int | None = None,
) -> nx.DiGraph:
    """Build a GLUE-style TF-target ``DiGraph`` from a HERTA edge table."""

    df = _as_edge_frame(edges)
    target_col = _resolve_target_col(df, target_col)
    graph_df = _filter_edge_frame(
        df,
        tf_col=tf_col,
        target_col=target_col,
        score_col=score_col,
        cell_type=cell_type,
        selected_tfs=selected_tfs,
        min_score=min_score,
        top_k_edges=top_k_edges,
        top_k_targets_per_tf=top_k_targets_per_tf,
    )
    graph = nx.DiGraph()
    attr_cols = [
        col
        for col in [score_col, weight_col, sign_col, evidence_col, "cell_type", "n_mediator_peaks", "mediator_peaks"]
        if col is not None and col in graph_df.columns
    ]
    for row in graph_df.itertuples(index=False):
        tf = str(getattr(row, tf_col))
        target = str(getattr(row, target_col))
        attrs = {col: _graphml_scalar(getattr(row, col)) for col in attr_cols}
        attrs = {key: value for key, value in attrs.items() if value is not None}
        if score_col in attrs and "weight" not in attrs:
            attrs["weight"] = attrs[score_col]
        graph.add_edge(tf, target, **attrs)
    nx.set_node_attributes(graph, "target", name="type")
    for tf in graph_df[tf_col].astype(str).unique():
        if tf in graph:
            graph.nodes[tf]["type"] = "TF"
    return graph


def filter_tf_target_graph(
    graph_or_edges: nx.DiGraph | pd.DataFrame | str | Path,
    tf_col: str = "tf",
    target_col: str | None = None,
    score_col: str = "score",
    cell_type: str | None = None,
    selected_tfs: list[str] | set[str] | tuple[str, ...] | None = None,
    min_score: float | None = None,
    top_k_edges: int | None = None,
    top_k_targets_per_tf: int | None = None,
) -> nx.DiGraph:
    """Filter a TF-target graph or edge table and return a ``DiGraph`` copy."""

    if isinstance(graph_or_edges, nx.DiGraph):
        df = nx.to_pandas_edgelist(graph_or_edges, source=tf_col, target=target_col or "target")
        target_col = target_col or "target"
        filtered = build_tf_target_graph(
            df,
            tf_col=tf_col,
            target_col=target_col,
            score_col=score_col if score_col in df.columns else "weight",
            cell_type=cell_type,
            selected_tfs=selected_tfs,
            min_score=min_score,
            top_k_edges=top_k_edges,
            top_k_targets_per_tf=top_k_targets_per_tf,
        )
        for node, attrs in graph_or_edges.nodes(data=True):
            if node in filtered:
                filtered.nodes[node].update(attrs)
        return filtered
    return build_tf_target_graph(
        graph_or_edges,
        tf_col=tf_col,
        target_col=target_col,
        score_col=score_col,
        cell_type=cell_type,
        selected_tfs=selected_tfs,
        min_score=min_score,
        top_k_edges=top_k_edges,
        top_k_targets_per_tf=top_k_targets_per_tf,
    )


def plot_tf_target_graph(
    graph_or_edges: nx.DiGraph | pd.DataFrame | str | Path,
    ax: plt.Axes | None = None,
    layout: str = "graphviz",
    with_labels: bool = True,
    figsize: tuple[float, float] = (10, 10),
    node_size: int = 650,
    arrows: bool = True,
    title: str | None = None,
    **filter_kwargs,
) -> plt.Axes:
    """Plot a TF-target network using GLUE's simple networkx style."""

    graph = graph_or_edges if isinstance(graph_or_edges, nx.DiGraph) else build_tf_target_graph(graph_or_edges, **filter_kwargs)
    if isinstance(graph_or_edges, nx.DiGraph) and filter_kwargs:
        graph = filter_tf_target_graph(graph, **filter_kwargs)
    pos = _tf_target_layout(graph, layout=layout)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    node_colors = ["#E45756" if graph.nodes[node].get("type") == "TF" else "#54A24B" for node in graph.nodes]
    widths = [max(0.8, 3.0 * float(data.get("weight", data.get("score", 1.0)))) for _, _, data in graph.edges(data=True)]
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=node_size,
        width=widths,
        edge_color="#4C78A8",
        font_size=9,
        arrows=arrows,
        arrowsize=14,
    )
    ax.set_title(title or "TF-target gene network")
    return ax


def export_tf_target_graph(
    graph_or_edges: nx.DiGraph | pd.DataFrame | str | Path,
    path: str | Path,
    **filter_kwargs,
) -> nx.DiGraph:
    """Export a TF-target network to GraphML and return the exported graph."""

    graph = graph_or_edges if isinstance(graph_or_edges, nx.DiGraph) else build_tf_target_graph(graph_or_edges, **filter_kwargs)
    if isinstance(graph_or_edges, nx.DiGraph) and filter_kwargs:
        graph = filter_tf_target_graph(graph, **filter_kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)
    return graph


def build_egrn_graph(
    paths: pd.DataFrame | str | Path,
    *,
    cell_type: str | None = None,
    score_col: str = "path_score",
    min_score: float | None = None,
    top_k_paths: int | None = 100,
) -> nx.DiGraph:
    """Build a three-layer TF→peak→target eGRN graph from inferred paths."""

    frame = _as_edge_frame(paths)
    required = {"tf", "peak", "target_gene", score_col}
    if missing := required.difference(frame.columns):
        raise ValueError(f"eGRN paths are missing required columns: {sorted(missing)}")
    if cell_type is not None:
        if "cell_type" not in frame:
            raise ValueError("cell_type filtering requires a 'cell_type' column.")
        frame = frame[frame["cell_type"].astype(str) == str(cell_type)]
    frame = frame.copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    if frame[score_col].isna().any() or not np.isfinite(
        frame[score_col].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{score_col} must contain finite values.")
    if min_score is not None:
        frame = frame[frame[score_col] >= float(min_score)]
    frame = frame.sort_values(score_col, ascending=False)
    if top_k_paths is not None:
        if top_k_paths <= 0:
            raise ValueError("top_k_paths must be positive.")
        frame = frame.head(int(top_k_paths))

    graph = nx.DiGraph()
    for row in frame.itertuples(index=False):
        tf = str(row.tf)
        peak = str(row.peak)
        gene = str(row.target_gene)
        tf_node = f"tf::{tf}"
        peak_node = f"peak::{peak}"
        gene_node = f"gene::{gene}"
        graph.add_node(tf_node, type="TF", label=tf)
        graph.add_node(peak_node, type="peak", label=peak)
        graph.add_node(gene_node, type="target", label=gene)
        path_score = float(getattr(row, score_col))
        tp_score = float(
            getattr(row, "tf_peak_score", path_score)
        )
        pg_score = float(
            getattr(row, "peak_gene_score", path_score)
        )
        _add_or_update_egrn_edge(
            graph,
            tf_node,
            peak_node,
            weight=tp_score,
            relation="tf_peak",
            path_score=path_score,
        )
        _add_or_update_egrn_edge(
            graph,
            peak_node,
            gene_node,
            weight=pg_score,
            relation="peak_gene",
            path_score=path_score,
        )
    graph.graph["cell_type"] = "" if cell_type is None else str(cell_type)
    return graph


def _add_or_update_egrn_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    **attributes,
) -> None:
    if graph.has_edge(source, target):
        existing = graph[source][target]
        existing["weight"] = max(
            float(existing.get("weight", 0.0)),
            float(attributes["weight"]),
        )
        existing["path_score"] = max(
            float(existing.get("path_score", 0.0)),
            float(attributes["path_score"]),
        )
    else:
        graph.add_edge(source, target, **attributes)


def plot_egrn_graph(
    graph_or_paths: nx.DiGraph | pd.DataFrame | str | Path,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 7),
    with_labels: bool = True,
    title: str | None = None,
    **filter_kwargs,
) -> plt.Axes:
    """Plot a compact three-column TF/peak/target eGRN network."""

    graph = (
        graph_or_paths
        if isinstance(graph_or_paths, nx.DiGraph)
        else build_egrn_graph(graph_or_paths, **filter_kwargs)
    )
    node_groups = {
        node_type: [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("type") == node_type
        ]
        for node_type in ("TF", "peak", "target")
    }
    positions: dict[str, tuple[float, float]] = {}
    for x, node_type in zip((0.0, 0.5, 1.0), ("TF", "peak", "target")):
        nodes = sorted(
            node_groups[node_type],
            key=lambda node: str(graph.nodes[node].get("label", node)),
        )
        y_values = np.linspace(1.0, 0.0, max(len(nodes), 2))[: len(nodes)]
        positions.update(
            {node: (x, float(y)) for node, y in zip(nodes, y_values)}
        )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    colors = {
        "TF": "#E45756",
        "peak": "#F2CF5B",
        "target": "#54A24B",
    }
    for node_type, marker, size in (
        ("TF", "s", 220),
        ("peak", "D", 90),
        ("target", "o", 140),
    ):
        nodes = node_groups[node_type]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodes,
                node_color=colors[node_type],
                node_shape=marker,
                node_size=size,
                ax=ax,
            )
    widths = [
        0.6 + 2.4 * max(0.0, min(1.0, float(data.get("weight", 0.0))))
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        edge_color="#4C78A8",
        alpha=0.55,
        arrows=True,
        arrowsize=12,
        ax=ax,
    )
    if with_labels:
        labels = {
            node: str(attributes.get("label", node))
            for node, attributes in graph.nodes(data=True)
        }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=8,
            ax=ax,
        )
    ax.text(0.0, 1.08, "TF", ha="center", fontweight="bold")
    ax.text(0.5, 1.08, "peak", ha="center", fontweight="bold")
    ax.text(1.0, 1.08, "target gene", ha="center", fontweight="bold")
    context = graph.graph.get("cell_type", "")
    ax.set_title(title or (f"{context}: TF→peak→gene eGRN" if context else "TF→peak→gene eGRN"))
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.08, 1.13)
    ax.axis("off")
    return ax


def export_egrn_graph(
    graph_or_paths: nx.DiGraph | pd.DataFrame | str | Path,
    path: str | Path,
    **filter_kwargs,
) -> nx.DiGraph:
    """Export a three-layer eGRN graph to GraphML."""

    graph = (
        graph_or_paths
        if isinstance(graph_or_paths, nx.DiGraph)
        else build_egrn_graph(graph_or_paths, **filter_kwargs)
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, target)
    return graph


def top_tf_targets(
    edges: pd.DataFrame | str | Path,
    cell_type: str | None = None,
    top_tfs: int = 5,
    top_targets_per_tf: int = 10,
    score_col: str = "score",
) -> pd.DataFrame:
    """Return top target genes for the strongest TFs in one cell type or all cell types."""

    df = pd.read_parquet(edges) if isinstance(edges, str | Path) else edges.copy()
    required = {"tf", "gene", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"edges must contain {sorted(required)}.")
    if cell_type is not None:
        if "cell_type" not in df.columns:
            raise ValueError("cell_type filtering requires a 'cell_type' column.")
        df = df[df["cell_type"].astype(str) == str(cell_type)].copy()
    if df.empty:
        return df

    group_cols = ["cell_type", "tf"] if "cell_type" in df.columns else ["tf"]
    rank_group = ["cell_type"] if "cell_type" in df.columns else None
    tf_rank = df.groupby(group_cols, as_index=False)[score_col].sum()
    tf_rank = tf_rank.sort_values((["cell_type"] if rank_group else []) + [score_col], ascending=False)
    top_tf_table = tf_rank.groupby(rank_group, group_keys=False).head(top_tfs) if rank_group else tf_rank.head(top_tfs)
    selected = df.merge(top_tf_table[group_cols], on=group_cols, how="inner")
    return selected.sort_values(score_col, ascending=False).groupby(group_cols, group_keys=False).head(top_targets_per_tf).reset_index(drop=True)


def grn_target_heatmap(
    edges: pd.DataFrame | str | Path,
    cell_type: str | None = None,
    top_tfs: int = 5,
    top_targets_per_tf: int = 10,
    score_col: str = "score",
    title: str | None = None,
) -> plt.Axes:
    """Plot top TF-target scores as a heatmap."""

    selected = top_tf_targets(edges, cell_type, top_tfs, top_targets_per_tf, score_col)
    if selected.empty:
        raise ValueError("No GRN edges available for the requested cell type.")
    if cell_type is None and selected["cell_type"].nunique() > 1:
        selected = selected.copy()
        selected["tf_label"] = selected["cell_type"].astype(str) + " | " + selected["tf"].astype(str)
    else:
        selected = selected.assign(tf_label=selected["tf"].astype(str))

    heatmap_df = selected.pivot_table(index="tf_label", columns="gene", values=score_col, aggfunc="max", fill_value=0.0)
    fig_width = max(7, min(18, 0.36 * heatmap_df.shape[1] + 3))
    fig_height = max(3.5, min(14, 0.38 * heatmap_df.shape[0] + 1.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(heatmap_df, cmap="viridis", linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title(title or ("Top TF-target GRN scores" if cell_type is None else f"{cell_type}: top TF-target GRN scores"))
    ax.set_xlabel("target gene")
    ax.set_ylabel("TF")
    ax.tick_params(axis="x", labelrotation=65)
    return _save_or_return(fig, _safe_plot_name("grn_target_heatmap", cell_type))


def cell_type_grn_network(
    edges: pd.DataFrame | str | Path,
    cell_type: str,
    top_tfs: int = 5,
    top_targets_per_tf: int = 8,
    score_col: str = "score",
    title: str | None = None,
) -> plt.Axes:
    """Draw a compact TF-to-target GRN network for one cell type."""

    selected = top_tf_targets(edges, cell_type, top_tfs, top_targets_per_tf, score_col)
    if selected.empty:
        raise ValueError(f"No GRN edges available for cell type '{cell_type}'.")
    tfs = selected["tf"].drop_duplicates().astype(str).tolist()
    genes = selected["gene"].drop_duplicates().astype(str).tolist()
    tf_y = {name: y for name, y in zip(tfs, np.linspace(1, 0, max(len(tfs), 2))[: len(tfs)])}
    gene_y = {name: y for name, y in zip(genes, np.linspace(1, 0, max(len(genes), 2))[: len(genes)])}
    scores = selected[score_col].to_numpy(dtype=float)
    denom = max(scores.max() - scores.min(), 1e-8)

    fig_height = max(4.5, min(13, 0.28 * (len(tfs) + len(genes)) + 2.5))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    for row in selected.itertuples(index=False):
        score = float(getattr(row, score_col))
        weight = (score - scores.min()) / denom
        ax.plot(
            [0.12, 0.88],
            [tf_y[str(row.tf)], gene_y[str(row.gene)]],
            color="#4C78A8",
            linewidth=0.6 + 3.0 * weight,
            alpha=0.25 + 0.55 * weight,
            zorder=1,
        )

    ax.scatter([0.12] * len(tfs), [tf_y[name] for name in tfs], s=170, marker="s", color="#E45756", zorder=3)
    ax.scatter([0.88] * len(genes), [gene_y[name] for name in genes], s=95, marker="o", color="#54A24B", zorder=3)
    for name in tfs:
        ax.text(0.08, tf_y[name], name, ha="right", va="center", fontsize=9, fontweight="bold")
    for name in genes:
        ax.text(0.92, gene_y[name], name, ha="left", va="center", fontsize=8)

    ax.text(0.12, 1.06, "TF", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.88, 1.06, "target gene", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title or f"{cell_type}: top TF-target GRN")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.12)
    ax.axis("off")
    return _save_or_return(fig, _safe_plot_name("grn_network", cell_type))


def _as_edge_frame(edges: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(edges, str | Path):
        path = Path(edges)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    return edges.copy()


def _resolve_target_col(df: pd.DataFrame, target_col: str | None) -> str:
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"edges must contain target column '{target_col}'.")
        return target_col
    for candidate in ("target", "gene", "target_gene", "Target gene"):
        if candidate in df.columns:
            return candidate
    raise ValueError("edges must contain a target column, e.g. 'target' or 'gene'.")


def _filter_edge_frame(
    df: pd.DataFrame,
    tf_col: str,
    target_col: str,
    score_col: str,
    cell_type: str | None,
    selected_tfs: list[str] | set[str] | tuple[str, ...] | None,
    min_score: float | None,
    top_k_edges: int | None,
    top_k_targets_per_tf: int | None,
) -> pd.DataFrame:
    required = {tf_col, target_col}
    if min_score is not None or top_k_edges is not None or top_k_targets_per_tf is not None:
        required.add(score_col)
    if missing := required.difference(df.columns):
        raise ValueError(f"edges are missing required columns: {sorted(missing)}")

    out = df.copy()
    if cell_type is not None:
        if "cell_type" not in out.columns:
            raise ValueError("cell_type filtering requires a 'cell_type' column.")
        out = out[out["cell_type"].astype(str) == str(cell_type)]
    if selected_tfs is not None:
        selected = set(map(str, selected_tfs))
        out = out[out[tf_col].astype(str).isin(selected)]
    if min_score is not None:
        out = out[pd.to_numeric(out[score_col]) >= float(min_score)]
    if score_col in out.columns:
        out = out.sort_values(score_col, ascending=False)
    if top_k_targets_per_tf is not None:
        out = out.groupby(tf_col, sort=False, group_keys=False).head(int(top_k_targets_per_tf))
    if top_k_edges is not None:
        out = out.head(int(top_k_edges))
    return out.reset_index(drop=True)


def _graphml_scalar(value: object) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _tf_target_layout(graph: nx.DiGraph, layout: str = "graphviz") -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if layout == "graphviz":
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            return graphviz_layout(graph)
        except Exception:
            layout = "spring"
    if layout == "spring":
        return nx.spring_layout(graph, seed=0)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    raise ValueError("layout must be 'graphviz', 'spring', or 'kamada_kawai'.")


def _safe_plot_name(prefix: str, value: str | None) -> str:
    suffix = "all" if value is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return f"{prefix}_{suffix}.png"


def embedding_umap(
    embeddings: pd.DataFrame | str | Path,
    labels: pd.Series | list[str] | dict[str, str] | str | None = None,
    name_col: str = "name",
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 1,
    title: str | None = None,
    filename: str | None = None,
) -> plt.Axes:
    """Visualize trained node embeddings after 2D projection.

    Parameters
    ----------
    embeddings:
        A node embedding DataFrame or parquet path. The HERTA writer stores a
        ``name`` column followed by numeric embedding dimensions.
    labels:
        Optional labels used for coloring. Pass a column name in ``embeddings``,
        a sequence aligned to rows, or a mapping from node name to label.
    method:
        ``"umap"`` for Scanpy-style neighbor graph + UMAP, or ``"pca"`` for a
        deterministic PCA projection.
    """

    df = pd.read_parquet(embeddings) if isinstance(embeddings, str | Path) else embeddings.copy()
    if name_col not in df.columns:
        raise ValueError(f"Expected a '{name_col}' column in embeddings.")
    value_cols = [c for c in df.columns if c != name_col and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("No numeric embedding columns found.")

    x = df[value_cols].to_numpy(dtype=np.float32)
    method = method.lower()
    if method == "umap":
        try:
            import scanpy as sc
            from anndata import AnnData

            adata = AnnData(x)
            sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, max(2, x.shape[0] - 1)), use_rep="X", random_state=random_state)
            sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
            coords = adata.obsm["X_umap"]
            x_label, y_label = "UMAP1", "UMAP2"
        except Exception as exc:
            print(f"UMAP projection failed ({exc}); falling back to PCA.")
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=random_state).fit_transform(x)
        x_label, y_label = "PC1", "PC2"
    elif method != "umap":
        raise ValueError("method must be 'umap' or 'pca'.")

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], name_col: df[name_col].astype(str)})
    hue = None
    if labels is not None:
        hue = "label"
        if isinstance(labels, str):
            if labels not in df.columns:
                raise ValueError(f"Label column '{labels}' is not present in embeddings.")
            plot_df[hue] = df[labels].astype(str).to_numpy()
        elif isinstance(labels, dict):
            plot_df[hue] = plot_df[name_col].map(labels).fillna("unknown").astype(str)
        else:
            label_values = pd.Series(labels)
            if len(label_values) != len(plot_df):
                raise ValueError("labels must have the same length as embeddings.")
            plot_df[hue] = label_values.astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=settings.figure_params.get("figsize", (6, 5)))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue=hue,
        s=28,
        linewidth=0,
        alpha=0.88,
        palette="tab10",
        ax=ax,
    )
    ax.set_title(title or f"{method.upper()} of node embeddings")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.set_yticks([])
    if hue is not None:
        ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    return _save_or_return(fig, filename or f"{method}_embeddings.png")
