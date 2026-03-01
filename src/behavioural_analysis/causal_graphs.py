import os

import matplotlib.pyplot as plt
import networkx as nx


def main() -> None:
    os.makedirs("plots", exist_ok=True)
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("attendance", "engagement_index"),
            ("engagement_index", "burnout_score"),
            ("attendance", "burnout_score"),
            ("procrastination_score", "burnout_score"),
            ("sentiment_score", "burnout_score"),
            ("burnout_score", "dropout_status"),
            ("burnout_score", "time_to_dropout"),
        ]
    )

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="#AED6F1",
        arrowsize=20,
        font_size=9,
    )
    plt.title("Conceptual Causal DAG for Burnout and Dropout")
    plt.tight_layout()

    out_path = "plots/causal_dag.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved causal DAG to {out_path}")


if __name__ == "__main__":
    main()

