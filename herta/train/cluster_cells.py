"""Cluster Stage-1 cell embeddings without requiring biological labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from herta.train.egrn import RegulatoryStateResult


def cluster_cells(
    embeddings: RegulatoryStateResult | pd.DataFrame | np.ndarray | str | Path,
    output_dir: str | Path,
    config: dict | None = None,
    cell_types: np.ndarray | pd.Series | list[str] | None = None,
    *,
    representation_mode: str | None = None,
) -> pd.DataFrame:
    """Cluster either the primary regulatory state or baseline ``z_cell``."""

    config = config or {}
    cluster_cfg = config.get("clustering", config)
    if isinstance(embeddings, RegulatoryStateResult):
        frame = embeddings.matrix.copy()
        resolved_mode = representation_mode or "regulatory_state"
    elif isinstance(embeddings, str | Path):
        path = Path(embeddings)
        if path.suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix == ".npz":
            payload = np.load(path)
            frame = pd.DataFrame(
                payload["matrix"],
                index=payload["cell_ids"].astype(str),
                columns=payload["selected_tfs"].astype(str),
            )
        else:
            raise ValueError("Clustering input path must be parquet, CSV, or NPZ.")
        resolved_mode = representation_mode or str(
            cluster_cfg.get("representation", "z_cell")
        )
    elif isinstance(embeddings, pd.DataFrame):
        frame = embeddings.copy()
        resolved_mode = representation_mode or str(
            cluster_cfg.get("representation", "z_cell")
        )
    else:
        frame = pd.DataFrame(np.asarray(embeddings))
        resolved_mode = representation_mode or str(
            cluster_cfg.get("representation", "z_cell")
        )
    if resolved_mode not in {"regulatory_state", "z_cell"}:
        raise ValueError(
            "representation_mode must be 'regulatory_state' or 'z_cell'."
        )
    name_col = "name" if "name" in frame else None
    if name_col:
        names = frame[name_col].astype(str).tolist()
    elif not isinstance(frame.index, pd.RangeIndex):
        names = frame.index.astype(str).tolist()
    else:
        names = [f"cell_{i}" for i in range(len(frame))]
    values = frame.drop(columns=[name_col] if name_col else []).to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Clustering requires at least two cells.")
    if values.shape[1] == 0:
        raise ValueError(
            "Clustering input has no selected features; adjust TF filtering "
            "or use the z_cell baseline."
        )
    if not np.isfinite(values).all():
        raise ValueError("Clustering input contains non-finite values.")
    method = str(cluster_cfg.get("method", "leiden")).lower()
    if method == "leiden":
        import scanpy as sc
        from anndata import AnnData

        adata = AnnData(values)
        sc.pp.neighbors(
            adata,
            n_neighbors=min(int(cluster_cfg.get("n_neighbors", 20)), max(2, len(values) - 1)),
            use_rep="X",
            random_state=int(config.get("seed", 1)),
        )
        sc.tl.leiden(
            adata,
            resolution=float(cluster_cfg.get("resolution", 0.6)),
            random_state=int(config.get("seed", 1)),
            key_added="cluster",
        )
        labels = adata.obs["cluster"].astype(str).to_numpy()
    elif method in {"kmeans", "kmeans_auto"}:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        seed = int(config.get("seed", 1))
        if method == "kmeans_auto":
            k_min = max(2, int(cluster_cfg.get("k_min", 2)))
            k_max = min(int(cluster_cfg.get("k_max", 20)), len(values) - 1)
            if k_min > k_max:
                raise ValueError("kmeans_auto requires 2 <= k_min <= k_max < n_cells.")
            sample_size = min(
                int(cluster_cfg.get("selection_sample_size", 2000)), len(values)
            )
            candidates: list[tuple[float, int, np.ndarray]] = []
            for n_clusters in range(k_min, k_max + 1):
                candidate = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=seed,
                    n_init=10,
                    batch_size=min(1024, len(values)),
                ).fit_predict(values)
                if np.unique(candidate).size < 2:
                    continue
                score = silhouette_score(
                    values,
                    candidate,
                    sample_size=sample_size,
                    random_state=seed,
                )
                candidates.append((float(score), n_clusters, candidate))
            if not candidates:
                raise RuntimeError(
                    "kmeans_auto could not produce at least two non-empty clusters."
                )
            selection_score, n_clusters, selected = max(
                candidates, key=lambda item: (item[0], -item[1])
            )
            labels = selected.astype(str)
        else:
            n_clusters = int(cluster_cfg.get("n_clusters", 2))
            labels = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=int(cluster_cfg.get("n_init", 50)),
            ).fit_predict(values).astype(str)
    else:
        raise ValueError(
            "clustering.method must be 'leiden', 'kmeans', or 'kmeans_auto'."
        )

    result = pd.DataFrame(
        {
            "cell_id": names,
            "cluster": labels,
            "clustering_input": resolved_mode,
        }
    )
    if method == "kmeans_auto":
        result["selected_n_clusters"] = int(n_clusters)
        result["cluster_selection_silhouette"] = float(selection_score)
    if bool(cluster_cfg.get("use_soft_assignments", False)):
        cluster_names = sorted(pd.unique(labels).astype(str))
        centroids = np.stack([values[np.asarray(labels).astype(str) == name].mean(axis=0) for name in cluster_names])
        squared_distance = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        temperature = float(cluster_cfg.get("soft_assignment_temperature", 1.0))
        if temperature <= 0:
            raise ValueError("soft_assignment_temperature must be positive.")
        logits = -squared_distance / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for index, name in enumerate(cluster_names):
            result[f"q_{name}"] = probabilities[:, index]
    if cell_types is not None:
        if len(cell_types) != len(result):
            raise ValueError("cell_types must align with the cell embeddings.")
        result["cell_type"] = pd.Series(cell_types).astype(str).to_numpy()
    output_dir = Path(output_dir)
    target = output_dir / "clustering"
    target.mkdir(parents=True, exist_ok=True)
    if resolved_mode == "regulatory_state":
        cluster_path = target / "regulatory_state_clusters.parquet"
        marker_path = target / "regulatory_state_cluster_markers.parquet"
    else:
        cluster_path = target / "cell_clusters.parquet"
        marker_path = target / "cluster_markers.parquet"
    result.to_parquet(cluster_path, index=False)
    pd.DataFrame(columns=["cluster", "gene", "score"]).to_parquet(
        marker_path,
        index=False,
    )
    return result
