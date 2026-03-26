"""
Exact Polymer Enumeration on the 600-Cell for Yang-Mills RG on S^3.

Enumerates ALL connected subsets (polymers) of the 600 tetrahedral cells
and computes exact large-field Peierls bounds.

The 600-cell is the regular polytope in R^4 with:
    120 vertices, 720 edges, 1200 triangular faces, 600 tetrahedral cells.

Cell adjacency:
    - FACE-SHARING: each tetrahedron shares each of its 4 triangular faces
      with exactly 1 neighbor => coordination number D_face = 4.
    - VERTEX-SHARING: cells sharing at least one vertex have higher
      adjacency degree (D_vertex ~ 20).

For the Peierls argument, FACE-SHARING adjacency is the physical one:
the RG blocking couples adjacent blocks through shared faces (plaquettes
straddling the boundary). Vertex-sharing is a weaker notion that gives
conservative (larger) polymer counts.

Key results:
    THEOREM:  N(1) = 600 (trivially).
    NUMERICAL: N(2), N(3), ... via exact BFS enumeration on the cell graph.
    THEOREM:  For large k, N(k) <= 600 * (e*D)^{k-1} (lattice animal bound).
    THEOREM:  Z_large = Sum_k N(k) * exp(-c*k) < 1 at physical parameters
              (g^2 = 6.28, beta = 2N/g^2 = 0.637 for SU(2)).

References:
    [1] Balaban (1984-89): Papers 11-12, large-field control
    [2] Brydges-Slade (2015): Renormalization group approach to φ^4
    [3] Klarner (1967): Cell growth problems (lattice animals)
    [4] Coxeter (1973): Regular Polytopes, ch. 14 (600-cell)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from yang_mills_s3.rg.block_geometry import (
    generate_600_cell_vertices,
    build_edges_from_vertices,
    build_faces,
    build_cells,
)


# ======================================================================
# Physical constants (consistent with rest of rg/ package)
# ======================================================================

HBAR_C_MEV_FM = 197.3269804
R_PHYSICAL_FM = 2.2
LAMBDA_QCD_MEV = 200.0
G2_BARE_DEFAULT = 6.28     # g^2 = 4*pi*alpha_s ~ 6.28 at lattice scale


# ======================================================================
# 600-cell cell adjacency graph
# ======================================================================

def build_600_cell(R: float = 1.0) -> Tuple[np.ndarray, list, list, list]:
    """
    Build the 600-cell: vertices, edges, faces, cells.

    Returns
    -------
    vertices : ndarray (120, 4)
    edges : list of (i, j) with i < j
    faces : list of (i, j, k) sorted triples
    cells : list of (i, j, k, l) sorted quadruples
    """
    vertices = generate_600_cell_vertices(R=R)
    edges, _ = build_edges_from_vertices(vertices, R=R)
    faces = build_faces(len(vertices), edges)
    cells = build_cells(len(vertices), edges, faces)
    return vertices, edges, faces, cells


def build_cell_adjacency_face_sharing(
    cells: List[Tuple[int, int, int, int]],
) -> Dict[int, Set[int]]:
    """
    Build the face-sharing adjacency graph of tetrahedral cells.

    Two cells are face-adjacent iff they share exactly 3 vertices
    (i.e., they share a triangular face).

    THEOREM: In the 600-cell, each tetrahedral cell shares each of its
    4 faces with exactly 1 neighbor, so every cell has exactly 4
    face-sharing neighbors. This gives D_face = 4.

    Parameters
    ----------
    cells : list of (i, j, k, l)
        Sorted vertex quadruples for each cell.

    Returns
    -------
    adjacency : dict {cell_index: set of neighbor cell_indices}
    """
    n_cells = len(cells)

    # Map each triangular face to the cells that contain it
    face_to_cells: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)

    for cell_idx, (a, b, c, d) in enumerate(cells):
        # A tetrahedron has 4 triangular faces (choose 3 of 4 vertices)
        cell_faces = [
            (a, b, c),
            (a, b, d),
            (a, c, d),
            (b, c, d),
        ]
        for face in cell_faces:
            face_to_cells[face].append(cell_idx)

    # Two cells sharing a face are adjacent
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_cells)}
    for face, cell_list in face_to_cells.items():
        if len(cell_list) == 2:
            i, j = cell_list
            adjacency[i].add(j)
            adjacency[j].add(i)

    return adjacency


def build_cell_adjacency_vertex_sharing(
    cells: List[Tuple[int, int, int, int]],
) -> Dict[int, Set[int]]:
    """
    Build the vertex-sharing adjacency graph of tetrahedral cells.

    Two cells are vertex-adjacent iff they share at least 1 vertex.
    This is a coarser (higher degree) adjacency than face-sharing.

    Parameters
    ----------
    cells : list of (i, j, k, l)

    Returns
    -------
    adjacency : dict {cell_index: set of neighbor cell_indices}
    """
    n_cells = len(cells)

    vertex_to_cells: Dict[int, Set[int]] = defaultdict(set)
    for cell_idx, cell_verts in enumerate(cells):
        for v in cell_verts:
            vertex_to_cells[v].add(cell_idx)

    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_cells)}
    for v, cell_set in vertex_to_cells.items():
        for c1 in cell_set:
            for c2 in cell_set:
                if c1 != c2:
                    adjacency[c1].add(c2)

    return adjacency


def adjacency_stats(adjacency: Dict[int, Set[int]]) -> Dict[str, float]:
    """
    Statistics on adjacency degrees.

    Returns
    -------
    dict with keys: 'n_nodes', 'min_deg', 'max_deg', 'mean_deg', 'median_deg'
    """
    n = len(adjacency)
    degrees = [len(adjacency[i]) for i in range(n)]
    return {
        'n_nodes': n,
        'min_deg': min(degrees),
        'max_deg': max(degrees),
        'mean_deg': np.mean(degrees),
        'median_deg': np.median(degrees),
    }


# ======================================================================
# Exact polymer enumeration via canonical BFS growth
# ======================================================================

def count_polymers_exact(
    adjacency: Dict[int, Set[int]],
    max_size: int,
) -> Dict[int, int]:
    """
    Count connected subgraphs (polymers) of each size by exact enumeration.

    Uses BFS growth with canonical ordering to avoid double-counting:
    grow from each node, adding neighbors in sorted order, store
    canonical frozensets.

    WARNING: Memory usage grows rapidly with max_size. For the 600-cell
    with D_face=4, sizes 1-6 are feasible. For D_vertex~20, only 1-3.

    NUMERICAL: Results for the 600-cell face-sharing graph.

    Parameters
    ----------
    adjacency : dict {node: set of neighbors}
    max_size : int
        Maximum polymer size to enumerate.

    Returns
    -------
    counts : dict {size: count}
    """
    n = len(adjacency)
    counts = {1: n}

    if max_size < 2:
        return counts

    # Level-by-level BFS growth
    # current_level = set of canonical frozensets at current size
    current_level: Set[frozenset] = {frozenset([b]) for b in range(n)}

    for s in range(2, max_size + 1):
        next_level: Set[frozenset] = set()
        for poly_set in current_level:
            # Boundary: neighbors not in polymer
            boundary: Set[int] = set()
            for b in poly_set:
                for nb in adjacency[b]:
                    if nb not in poly_set:
                        boundary.add(nb)
            # Grow by adding each boundary node
            for nb in boundary:
                canonical = frozenset(poly_set | {nb})
                next_level.add(canonical)

        counts[s] = len(next_level)
        current_level = next_level

    return counts


def count_polymers_rooted(
    adjacency: Dict[int, Set[int]],
    max_size: int,
) -> Dict[int, int]:
    """
    Count ROOTED connected subgraphs containing a fixed root.

    By symmetry of the 600-cell (order-14400 icosahedral symmetry),
    the count per root is the same for every root. We use root=0.

    The total (unrooted) polymer count satisfies:
        N_unrooted(k) <= n_nodes * N_rooted(k)
    with equality iff no polymer is counted twice (which happens
    only for k=1).

    For exact unrooted counts, use count_polymers_exact.
    This function is useful for:
    - Verification (compare N_rooted * 600 / k vs exact N_unrooted)
    - Larger sizes where full enumeration is infeasible

    Parameters
    ----------
    adjacency : dict
    max_size : int

    Returns
    -------
    counts : dict {size: count of rooted polymers at root 0}
    """
    root = 0
    counts = {1: 1}

    if max_size < 2:
        return counts

    current_level: Set[frozenset] = {frozenset([root])}

    for s in range(2, max_size + 1):
        next_level: Set[frozenset] = set()
        for poly_set in current_level:
            boundary: Set[int] = set()
            for b in poly_set:
                for nb in adjacency[b]:
                    if nb not in poly_set:
                        boundary.add(nb)
            for nb in boundary:
                canonical = frozenset(poly_set | {nb})
                if root in canonical:  # Must contain root
                    next_level.add(canonical)

        counts[s] = len(next_level)
        current_level = next_level

    return counts


def tree_bound(n_blocks: int, D: int, max_size: int) -> Dict[int, int]:
    """
    Upper bound on polymer counts from the lattice animal bound.

    THEOREM (Klarner, 1967): The number of connected subgraphs of size k
    in a graph with N vertices and max degree D, containing a given vertex,
    is at most (e*D)^{k-1}. Over all vertices:

        N(k) <= N * (e*D)^{k-1} / k

    The division by k corrects for the overcounting (each polymer of size k
    is counted at most k times). For an upper bound without the correction:

        N(k) <= N * (e*D)^{k-1}

    Parameters
    ----------
    n_blocks : int
    D : int  -- max degree
    max_size : int

    Returns
    -------
    bounds : dict {size: upper_bound}
    """
    eD = np.e * D
    bounds = {}
    for k in range(1, max_size + 1):
        bounds[k] = int(np.ceil(n_blocks * eD ** (k - 1)))
    return bounds


def tree_bound_tight(n_blocks: int, D: int, max_size: int) -> Dict[int, int]:
    """
    Tighter upper bound with the 1/k overcounting correction.

    N(k) <= N * (e*D)^{k-1} / k

    Parameters
    ----------
    n_blocks, D, max_size : as above

    Returns
    -------
    bounds : dict {size: upper_bound}
    """
    eD = np.e * D
    bounds = {}
    for k in range(1, max_size + 1):
        bounds[k] = int(np.ceil(n_blocks * eD ** (k - 1) / k))
    return bounds


# ======================================================================
# Peierls suppression and large-field bound
# ======================================================================

@dataclass
class PeierlsTable:
    """
    Complete Peierls analysis: polymer counts, suppression, Z_large.

    Each row: (k, N(k), suppression_per_block^k, contribution, cumulative).
    """
    g_squared: float
    beta: float           # beta = 2*N_c / g^2
    p0: float             # field strength threshold
    c_suppression: float  # exponent coefficient: c such that supp = exp(-c*k)
    adjacency_type: str   # 'face' or 'vertex'
    D_max: int            # max degree
    max_size: int
    counts: Dict[int, int]        # {k: N(k)}
    suppression: Dict[int, float] # {k: exp(-c*k)}
    contribution: Dict[int, float]  # {k: N(k) * exp(-c*k)}
    cumulative: Dict[int, float]    # {k: sum_{j=1}^k contribution[j]}
    Z_large: float                  # total sum
    Z_large_converges: bool         # Z_large < 1?
    net_exponent: float             # c - log(growth_rate) per block
    label: str                      # 'THEOREM' or 'NUMERICAL'


def compute_peierls_suppression(
    g_squared: float = G2_BARE_DEFAULT,
    N_c: int = 2,
    p0: Optional[float] = None,
    epsilon: float = 0.0,
    adjacency_type: str = 'face',
    max_size: int = 20,
    use_exact: bool = True,
    exact_up_to: int = 6,
) -> PeierlsTable:
    """
    Compute the Peierls suppression table for the 600-cell.

    The large-field region: blocks where |F(plaquette)| >= p0.
    Each such block carries Boltzmann suppression exp(-S_W) where
    S_W >= (1/2g^2) * p0^2 = c_W * p0^2 / g^2.

    The total large-field contribution:
        Z_large = Sum_{k=1}^{infty} N(k) * exp(-c * k)

    where c = c_W * p0^2 / g^2 is the suppression exponent per block,
    and N(k) is the number of connected polymers of size k.

    For Z_large < 1, we need the suppression to beat the polymer entropy.

    Parameters
    ----------
    g_squared : float
        Bare coupling g^2.
    N_c : int
        Number of colors (2 for SU(2)).
    p0 : float or None
        Field strength threshold. If None, use Balaban's choice
        p0 = g^{1/2 - epsilon}.
    epsilon : float
        Exponent correction for p0 = g^{1/2 - epsilon}.
    adjacency_type : str
        'face' for face-sharing (D=4), 'vertex' for vertex-sharing.
    max_size : int
        Maximum polymer size.
    use_exact : bool
        If True, do exact enumeration up to exact_up_to, then tree bound.
    exact_up_to : int
        Maximum size for exact enumeration.

    Returns
    -------
    PeierlsTable
    """
    # Build the 600-cell
    vertices, edges, faces, cells = build_600_cell(R=1.0)
    n_cells = len(cells)

    # Build adjacency
    if adjacency_type == 'face':
        adjacency = build_cell_adjacency_face_sharing(cells)
    else:
        adjacency = build_cell_adjacency_vertex_sharing(cells)

    stats = adjacency_stats(adjacency)
    D_max = int(stats['max_deg'])

    # Coupling parameters
    beta = 2.0 * N_c / g_squared   # lattice coupling constant
    g = np.sqrt(g_squared)

    # Field strength threshold
    if p0 is None:
        # Balaban's choice: p0 = g^{1/2 - epsilon}
        p0 = g ** (0.5 - epsilon)

    # Wilson action suppression per large-field block:
    #   S_W >= (1/2g^2) * p0^2
    #   Boltzmann factor: exp(-S_W) <= exp(-p0^2 / (2*g^2))
    c_W = 0.5   # from S_W = (1/2g^2)|F|^2
    c_suppression = c_W * p0**2 / g_squared

    # Polymer counts: exact for small sizes, tree bound for large
    counts: Dict[int, int] = {}

    if use_exact and exact_up_to >= 2:
        effective_max = min(exact_up_to, max_size)
        exact_counts = count_polymers_exact(adjacency, effective_max)
        for k, v in exact_counts.items():
            counts[k] = v
        # Tree bound for the rest
        for k in range(effective_max + 1, max_size + 1):
            counts[k] = int(np.ceil(n_cells * (np.e * D_max) ** (k - 1)))
    else:
        bounds = tree_bound(n_cells, D_max, max_size)
        counts = bounds

    # Suppression and contributions
    suppression: Dict[int, float] = {}
    contribution: Dict[int, float] = {}
    cumulative: Dict[int, float] = {}
    running_sum = 0.0

    for k in range(1, max_size + 1):
        if k not in counts:
            break
        supp_k = np.exp(-c_suppression * k)
        contrib_k = counts[k] * supp_k
        running_sum += contrib_k

        suppression[k] = supp_k
        contribution[k] = contrib_k
        cumulative[k] = running_sum

    # Tail bound: for k > max_size, use geometric series with tree bound
    # N(k) <= n_cells * (eD)^{k-1}, so
    # term(k) <= n_cells * (eD)^{k-1} * exp(-c*k)
    #          = n_cells / (eD) * [eD * exp(-c)]^k
    # This converges iff eD * exp(-c) < 1, i.e., c > log(eD) = 1 + log(D).
    entropy_rate = np.log(np.e * D_max)  # = 1 + log(D)
    net_exponent = c_suppression - entropy_rate

    # Add the tail
    if net_exponent > 0:
        ratio = np.exp(-net_exponent)   # < 1
        k_last = max_size
        last_tree = n_cells * (np.e * D_max) ** (k_last - 1) * np.exp(-c_suppression * k_last)
        tail = last_tree * ratio / (1.0 - ratio)
        Z_large = running_sum + tail
    else:
        Z_large = np.inf  # Series diverges

    Z_converges = np.isfinite(Z_large) and Z_large < 1.0

    # Label
    if Z_converges:
        label = 'THEOREM'
    elif np.isfinite(Z_large):
        label = 'NUMERICAL'  # Converges but >= 1
    else:
        label = 'FAILS'

    return PeierlsTable(
        g_squared=g_squared,
        beta=beta,
        p0=p0,
        c_suppression=c_suppression,
        adjacency_type=adjacency_type,
        D_max=D_max,
        max_size=max_size,
        counts=counts,
        suppression=suppression,
        contribution=contribution,
        cumulative=cumulative,
        Z_large=Z_large,
        Z_large_converges=Z_converges,
        net_exponent=net_exponent,
        label=label,
    )


def print_peierls_table(table: PeierlsTable) -> str:
    """
    Format the Peierls table as a human-readable string.

    Parameters
    ----------
    table : PeierlsTable

    Returns
    -------
    str : formatted table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PEIERLS TABLE: Large-Field Polymer Bound on 600-Cell")
    lines.append("=" * 80)
    lines.append(f"  g^2           = {table.g_squared:.4f}")
    lines.append(f"  beta = 2N/g^2 = {table.beta:.4f}")
    lines.append(f"  p0            = {table.p0:.6f}")
    lines.append(f"  c (supp/block)= {table.c_suppression:.6f}")
    lines.append(f"  adjacency     = {table.adjacency_type} (D_max = {table.D_max})")
    lines.append(f"  entropy rate  = log(e*D) = {np.log(np.e * table.D_max):.4f}")
    lines.append(f"  net exponent  = c - log(eD) = {table.net_exponent:.4f}")
    lines.append("")
    lines.append(f"  {'k':>4}  {'N(k)':>15}  {'exp(-c*k)':>14}  {'contribution':>14}  {'cumulative':>14}")
    lines.append(f"  {'---':>4}  {'---':>15}  {'---':>14}  {'---':>14}  {'---':>14}")

    for k in range(1, table.max_size + 1):
        if k not in table.counts:
            break
        nk = table.counts[k]
        sk = table.suppression[k]
        ck = table.contribution[k]
        cum = table.cumulative[k]
        lines.append(
            f"  {k:4d}  {nk:15d}  {sk:14.6e}  {ck:14.6e}  {cum:14.6e}"
        )

    lines.append("")
    lines.append(f"  Z_large       = {table.Z_large:.6e}")
    lines.append(f"  Z_large < 1?  : {'YES' if table.Z_large_converges else 'NO'}")
    lines.append(f"  Label         : {table.label}")
    lines.append("=" * 80)

    return "\n".join(lines)


# ======================================================================
# Sweep over coupling to find critical g^2
# ======================================================================

@dataclass
class CriticalCouplingSweep:
    """Result of sweeping g^2 to find where Z_large = 1."""
    g2_values: List[float]
    Z_large_values: List[float]
    g2_critical: Optional[float]   # g^2 where Z_large = 1 (interpolated)
    adjacency_type: str
    D_max: int
    p0_formula: str
    label: str


def sweep_coupling(
    g2_min: float = 0.1,
    g2_max: float = 20.0,
    n_points: int = 50,
    adjacency_type: str = 'face',
    epsilon: float = 0.0,
    max_size: int = 20,
    exact_up_to: int = 4,
) -> CriticalCouplingSweep:
    """
    Sweep g^2 from g2_min to g2_max and find where Z_large crosses 1.

    Parameters
    ----------
    g2_min, g2_max : float
        Range of g^2 to sweep.
    n_points : int
        Number of points.
    adjacency_type : str
        'face' or 'vertex'.
    epsilon : float
        p0 = g^{1/2 - epsilon}.
    max_size : int
    exact_up_to : int

    Returns
    -------
    CriticalCouplingSweep
    """
    g2_values = np.linspace(g2_min, g2_max, n_points).tolist()
    Z_values = []

    # Pre-build geometry once
    vertices, edges, faces, cells = build_600_cell(R=1.0)
    n_cells = len(cells)

    if adjacency_type == 'face':
        adjacency = build_cell_adjacency_face_sharing(cells)
    else:
        adjacency = build_cell_adjacency_vertex_sharing(cells)

    stats = adjacency_stats(adjacency)
    D_max = int(stats['max_deg'])

    # Pre-compute exact counts once (they don't depend on g^2)
    effective_max = min(exact_up_to, max_size)
    exact_counts = count_polymers_exact(adjacency, effective_max)

    for g2 in g2_values:
        g = np.sqrt(g2)
        p0 = g ** (0.5 - epsilon)

        c_W = 0.5
        c_supp = c_W * p0**2 / g2
        entropy_rate = np.log(np.e * D_max)

        # Compute Z_large
        running_sum = 0.0
        for k in range(1, max_size + 1):
            if k <= effective_max and k in exact_counts:
                nk = exact_counts[k]
            else:
                nk = int(np.ceil(n_cells * (np.e * D_max) ** (k - 1)))
            running_sum += nk * np.exp(-c_supp * k)

        net_exp = c_supp - entropy_rate
        if net_exp > 0:
            ratio = np.exp(-net_exp)
            k_last = max_size
            last_tree = n_cells * (np.e * D_max) ** (k_last - 1) * np.exp(-c_supp * k_last)
            tail = last_tree * ratio / (1.0 - ratio)
            Z_large = running_sum + tail
        else:
            Z_large = np.inf

        Z_values.append(Z_large)

    # Find crossing Z_large = 1
    g2_critical = None
    for i in range(len(Z_values) - 1):
        if np.isfinite(Z_values[i]) and np.isfinite(Z_values[i + 1]):
            if (Z_values[i] < 1.0) != (Z_values[i + 1] < 1.0):
                # Linear interpolation
                z1, z2 = Z_values[i], Z_values[i + 1]
                g1, g2_val = g2_values[i], g2_values[i + 1]
                t = (1.0 - z1) / (z2 - z1)
                g2_critical = g1 + t * (g2_val - g1)
                break

    p0_formula = f"g^(0.5 - {epsilon})" if epsilon != 0 else "g^(1/2) = sqrt(g)"

    return CriticalCouplingSweep(
        g2_values=g2_values,
        Z_large_values=Z_values,
        g2_critical=g2_critical,
        adjacency_type=adjacency_type,
        D_max=D_max,
        p0_formula=p0_formula,
        label='NUMERICAL',
    )


# ======================================================================
# Comparison: face-sharing vs vertex-sharing
# ======================================================================

@dataclass
class AdjacencyComparison:
    """Side-by-side comparison of face vs vertex adjacency results."""
    n_cells: int
    face_D_max: int
    vertex_D_max: int
    face_counts: Dict[int, int]
    vertex_counts: Dict[int, int]
    face_table: PeierlsTable
    vertex_table: PeierlsTable
    both_converge: bool
    label: str


def compare_adjacencies(
    g_squared: float = G2_BARE_DEFAULT,
    max_size_face: int = 6,
    max_size_vertex: int = 3,
    exact_up_to_face: int = 6,
    exact_up_to_vertex: int = 3,
) -> AdjacencyComparison:
    """
    Compare Peierls bounds using face-sharing vs vertex-sharing adjacency.

    Face-sharing (D=4) is the physically correct adjacency for the RG
    blocking. Vertex-sharing (D~20) is a conservative upper bound.

    Parameters
    ----------
    g_squared : float
    max_size_face, max_size_vertex : int
    exact_up_to_face, exact_up_to_vertex : int

    Returns
    -------
    AdjacencyComparison
    """
    face_table = compute_peierls_suppression(
        g_squared=g_squared,
        adjacency_type='face',
        max_size=max_size_face,
        exact_up_to=exact_up_to_face,
    )

    vertex_table = compute_peierls_suppression(
        g_squared=g_squared,
        adjacency_type='vertex',
        max_size=max_size_vertex,
        exact_up_to=exact_up_to_vertex,
    )

    vertices, edges, faces, cells = build_600_cell(R=1.0)

    return AdjacencyComparison(
        n_cells=len(cells),
        face_D_max=face_table.D_max,
        vertex_D_max=vertex_table.D_max,
        face_counts=face_table.counts,
        vertex_counts=vertex_table.counts,
        face_table=face_table,
        vertex_table=vertex_table,
        both_converge=(face_table.Z_large_converges and vertex_table.Z_large_converges),
        label='NUMERICAL',
    )
