"""Contact network generation for epidemic spread."""

import networkx as nx


def make_contact_network(n: int, kind: str = "ws", seed: int = 42) -> nx.Graph:
    """
    Build a contact network.

    Parameters
    ----------
    n : int
        Number of individuals (nodes).
    kind : str
        'er' (Erdos-Renyi), 'ws' (Watts-Strogatz), or 'ba' (Barabasi-Albert).
    seed : int
        Random seed for reproducibility.
    """
    if kind == "er":
        return nx.erdos_renyi_graph(n, p=0.15, seed=seed)
    if kind == "ws":
        k = min(4, n - 1) if n > 1 else 1
        return nx.watts_strogatz_graph(n, k=k, p=0.10, seed=seed)
    if kind == "ba":
        m = min(2, n - 1) if n > 1 else 1
        return nx.barabasi_albert_graph(n, m=m, seed=seed)
    raise ValueError(f"Unknown network kind: {kind!r}. Use 'er', 'ws', or 'ba'.")


def network_summary(G: nx.Graph) -> dict:
    """Return basic network statistics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": sum(degrees) / n if n else 0.0,
        "is_connected": nx.is_connected(G) if n > 0 else True,
    }
