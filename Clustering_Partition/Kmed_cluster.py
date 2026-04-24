import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def load_distance_matrix(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    D = df.to_numpy(dtype=float)
    node_ids = [int(x) for x in df.index.tolist()]

    if D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {D.shape}.")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-9):
        raise ValueError("Diagonal entries must be zero.")
    if np.any(D < -1e-12):
        raise ValueError("Distance matrix contains negative entries.")

    return D, node_ids


def pam_k_medoids(D, k, random_state=0, max_iter=200):
    """
    PAM (Partitioning Around Medoids)
    Obj:
        min sum_i d(i, medoid_of_cluster(i))
    """
    n = D.shape[0]
    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, size=k, replace=False)

    labels = np.argmin(D[:, medoids], axis=1)
    total_cost = float(D[np.arange(n), medoids[labels]].sum())

    for _ in range(max_iter):
        improved = False
        best_cost = total_cost
        best_medoids = medoids.copy()
        best_labels = labels.copy()

        medoid_set = set(medoids.tolist())
        non_medoids = [i for i in range(n) if i not in medoid_set]

        for medoid_pos, medoid_idx in enumerate(medoids):
            for cand in non_medoids:
                trial_medoids = medoids.copy()
                trial_medoids[medoid_pos] = cand
                trial_labels = np.argmin(D[:, trial_medoids], axis=1)
                trial_cost = float(D[np.arange(n), trial_medoids[trial_labels]].sum())

                if trial_cost < best_cost - 1e-12:
                    best_cost = trial_cost
                    best_medoids = trial_medoids
                    best_labels = trial_labels
                    improved = True

        if not improved:
            break

        medoids = best_medoids
        labels = best_labels
        total_cost = best_cost

    medoids = np.sort(medoids)
    labels = np.argmin(D[:, medoids], axis=1)
    total_cost = float(D[np.arange(D.shape[0]), medoids[labels]].sum())
    return medoids, labels, total_cost


def choose_best_k(D, k_min=4, k_max=16, random_state=0):
    rows = []
    for k in range(k_min, k_max + 1):
        medoids, labels, total_cost = pam_k_medoids(D, k, random_state=random_state)
        score = silhouette_score(D, labels, metric="precomputed")
        rows.append({
            "k": k,
            "silhouette": float(score),
            "within_cluster_cost": float(total_cost),
            "medoids_0based": medoids.tolist(),
        })
    return pd.DataFrame(rows).sort_values(
        by=["silhouette", "within_cluster_cost"],
        ascending=[False, True],
    )


def cluster_summary(D, node_ids, medoids, labels, top_r=2):
    node_rows = []
    cluster_rows = []

    for c, medoid_idx in enumerate(medoids):
        members = np.where(labels == c)[0]
        member_ids = [node_ids[i] for i in members]
        medoid_node = node_ids[medoid_idx]

        sub_D = D[np.ix_(members, members)]
        score = sub_D.sum(axis=1)
        order = np.argsort(score)
        top_members = members[order[: min(top_r, len(members))]]
        top_member_ids = [node_ids[i] for i in top_members]

        cluster_rows.append({
            "cluster_label": int(c),
            "cluster_size": int(len(members)),
            "medoid_node": int(medoid_node),
            "members": member_ids,
            "top_representatives": top_member_ids,
        })

        for i in members:
            node_rows.append({
                "node_id": int(node_ids[i]),
                "cluster_label": int(c),
                "assigned_medoid": int(medoid_node),
                "distance_to_medoid": float(D[i, medoid_idx]),
            })

    return pd.DataFrame(node_rows), pd.DataFrame(cluster_rows)


if __name__ == "__main__":
    csv_path = "TN_distance_matrix.csv"
    D, node_ids = load_distance_matrix(csv_path)

    # 1) choose best k
    score_df = choose_best_k(D, k_min=2, k_max=10, random_state=42)
    print(score_df)

    best_k = int(score_df.iloc[0]["k"])
    print("best_k =", best_k)

    # 2) Use best_k to cluster
    medoids, labels, total_cost = pam_k_medoids(D, best_k, random_state=42)
    print("medoids =", [node_ids[i] for i in medoids])
    print("total_cost =", total_cost)

    # 3) output results
    node_df, cluster_df = cluster_summary(D, node_ids, medoids, labels, top_r=2)
    print(cluster_df)

    node_df.to_csv("node_partition.csv", index=False)
    cluster_df.to_csv("cluster_summary.csv", index=False)
