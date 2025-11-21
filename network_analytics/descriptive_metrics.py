#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
network_analytics/descriptive_metrics.py

Descriptive Metrics Module for Actor-Event Network Analysis
Phase 1: Centrality & Graph Characterization

This module calculates node-level and graph-level network metrics to
understand network structure, identify key actors, and assess network cohesion.

Author: Network Analytics System
Version: 1.0
Date: 2025-11-21
"""

from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
import numpy as np
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# CLASS: DescriptiveMetricsCalculator
# ============================================================================

class DescriptiveMetricsCalculator:
    """
    Calculate node-level and graph-level network metrics.

    This class provides methods for computing centrality measures and graph
    statistics used to characterize network structure and identify key actors.

    Attributes:
        graph (nx.Graph): NetworkX graph object
        db (Optional[sqlite3.Connection]): SQLite connection for storage
        node_count (int): Number of nodes in graph
        edge_count (int): Number of edges in graph
        _metrics_cache (Dict): Cache for calculated metrics

    Example:
        >>> G = nx.Graph()
        >>> G.add_edges_from([('A', 'B'), ('B', 'C')])
        >>> calc = DescriptiveMetricsCalculator(G)
        >>> centrality = calc.degree_centrality()
        >>> print(centrality)
        {'A': 0.5, 'B': 1.0, 'C': 0.5}
    """

    def __init__(
        self, 
        graph: nx.Graph,
        db_connection: Optional[sqlite3.Connection] = None
    ):
        """
        Initialize metrics calculator.

        Args:
            graph: NetworkX graph object (DiGraph or Graph)
            db_connection: Optional SQLite connection for storing results

        Raises:
            ValueError: If graph is empty
            TypeError: If graph is not a NetworkX Graph

        Example:
            >>> import networkx as nx
            >>> G = nx.complete_graph(5, create_using=nx.Graph)
            >>> calc = DescriptiveMetricsCalculator(G)
        """
        self.validate_graph(graph)
        self.graph = graph
        self.db = db_connection
        self.node_count = len(graph)
        self.edge_count = graph.number_of_edges()
        self._metrics_cache = {}

        logger.info(
            f"Initialized DescriptiveMetricsCalculator: "
            f"nodes={self.node_count}, edges={self.edge_count}, "
            f"graph_type={type(graph).__name__}"
        )

    # ========================================================================
    # VALIDATION & UTILITY METHODS
    # ========================================================================

    @staticmethod
    def validate_graph(graph: Union[nx.Graph, nx.DiGraph]) -> None:
        """
        Validate that graph is a NetworkX Graph and non-empty.

        Args:
            graph: Object to validate

        Raises:
            TypeError: If graph is not a NetworkX Graph or DiGraph
            ValueError: If graph is empty
        """
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise TypeError(
                f"Expected nx.Graph or nx.DiGraph, got {type(graph).__name__}"
            )

        if len(graph) == 0:
            raise ValueError("Graph is empty (contains 0 nodes)")

    def _get_node_list(
        self, 
        node_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get node list for calculation.

        Args:
            node_list: Specific nodes to calculate. If None, use all nodes.

        Returns:
            List of node identifiers
        """
        if node_list is None:
            return list(self.graph.nodes())

        # Validate all requested nodes exist
        for node in node_list:
            if node not in self.graph:
                logger.warning(f"Node {node} not found in graph")

        return node_list

    def _validate_centrality_output(
        self, 
        centrality_dict: Dict[str, float],
        expected_range: Tuple[float, float] = (0.0, 1.0)
    ) -> bool:
        """
        Validate that centrality values are in expected range.

        Args:
            centrality_dict: Dictionary of centrality values
            expected_range: Tuple of (min, max) values

        Returns:
            True if valid, False otherwise
        """
        min_val, max_val = expected_range

        for node, value in centrality_dict.items():
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"Node {node} centrality {value} outside range [{min_val}, {max_val}]"
                )
                return False

        return True

    # ========================================================================
    # METHOD 1.1: DEGREE CENTRALITY
    # ========================================================================

    def degree_centrality(
        self, 
        node_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate degree centrality for all or specified nodes.

        Degree centrality measures the direct connectedness of a node.
        For node v with degree deg(v) in a graph with n nodes:

            DC(v) = deg(v) / (n - 1)

        Range: [0, 1]
            - 0: Isolated node (no connections)
            - 1: Connected to all other nodes

        Args:
            node_list: Specific nodes to calculate. If None, calculate all.

        Returns:
            Dict mapping node_id -> centrality_value (0.0-1.0)

        Raises:
            RuntimeError: If calculation fails

        Use Cases:
            - Identify most connected/popular actors
            - Find key participants in network
            - Baseline measure of actor importance

        Example:
            >>> G = nx.Graph()
            >>> G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
            >>> calc = DescriptiveMetricsCalculator(G)
            >>> dc = calc.degree_centrality()
            >>> print(dc['A'])  # A connected to 2 others = 2/2 = 1.0
            1.0

        Time Complexity: O(n + m) where n = nodes, m = edges
        Space Complexity: O(n)
        """
        try:
            logger.debug("Starting degree centrality calculation")

            # Get node list
            nodes = self._get_node_list(node_list)

            if not nodes:
                raise ValueError("No nodes to calculate")

            # Handle single-node case
            if self.node_count == 1:
                logger.debug("Single-node graph: degree centrality = 0")
                return {list(self.graph.nodes())[0]: 0.0}

            # Calculate using NetworkX
            centrality = nx.degree_centrality(self.graph)

            # Filter to requested nodes if specified
            if node_list is not None:
                centrality = {n: centrality[n] for n in nodes if n in centrality}

            # Validate output
            self._validate_centrality_output(centrality, (0.0, 1.0))

            logger.info(
                f"Degree centrality calculated: {len(centrality)} nodes, "
                f"range=[{min(centrality.values()):.4f}, {max(centrality.values()):.4f}]"
            )

            return centrality

        except Exception as e:
            logger.error(f"Degree centrality calculation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to calculate degree centrality: {str(e)}"
            )

    # ========================================================================
    # METHOD 1.2: BETWEENNESS CENTRALITY (Preview for Phase 1)
    # ========================================================================

    def betweenness_centrality(
        self, 
        normalized: bool = True,
        node_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all or specified nodes.

        Betweenness centrality measures how often a node appears on shortest
        paths between other pairs of nodes. Identifies key intermediaries.

        For node v in a graph with n nodes:

            BC(v) = 2 * Σ(σ_st(v) / σ_st) / ((n-1)(n-2))

        where:
            σ_st = number of shortest paths from s to t
            σ_st(v) = shortest paths through v

        Range: [0, 1] (when normalized)
            - 0: Node not on any shortest paths
            - 1: Node lies on all shortest paths

        Args:
            normalized: If True, normalize by (n-1)(n-2)/2
            node_list: Specific nodes to calculate

        Returns:
            Dict mapping node_id -> centrality_value

        Raises:
            RuntimeError: If calculation fails

        Use Cases:
            - Identify bridges and bottlenecks
            - Find key intermediaries in network
            - Information flow control points

        Time Complexity: O(n³) for unweighted graphs (Brandes' algorithm)
        Space Complexity: O(n²)
        """
        try:
            logger.debug("Starting betweenness centrality calculation")

            nodes = self._get_node_list(node_list)

            # Use Brandes' algorithm via NetworkX
            centrality = nx.betweenness_centrality(
                self.graph,
                normalized=normalized
            )

            # Filter if needed
            if node_list is not None:
                centrality = {n: centrality[n] for n in nodes if n in centrality}

            self._validate_centrality_output(centrality, (0.0, 1.0))

            logger.info(
                f"Betweenness centrality calculated: {len(centrality)} nodes"
            )

            return centrality

        except Exception as e:
            logger.error(f"Betweenness centrality calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate betweenness centrality: {str(e)}")

    # ========================================================================
    # METHOD 1.3: EIGENVECTOR CENTRALITY (Preview for Phase 1)
    # ========================================================================

    def eigenvector_centrality(
        self, 
        max_iter: int = 100,
        tol: float = 1e-6,
        node_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate eigenvector centrality for all or specified nodes.

        Eigenvector centrality measures a node's influence based on its
        connections to other influential nodes. Nodes with high centrality
        are connected to other high-centrality nodes.

        Solved iteratively via power iteration method.

        Range: [0, 1]

        Args:
            max_iter: Maximum iterations for power iteration
            tol: Convergence tolerance
            node_list: Specific nodes to calculate

        Returns:
            Dict mapping node_id -> centrality_value

        Raises:
            RuntimeError: If calculation fails
            ValueError: If disconnected graph and cannot find dominant eigenvector

        Use Cases:
            - Identify most influential actors
            - Find hierarchical leaders
            - Power structure analysis

        Note:
            - Works best on connected graphs
            - For disconnected graphs, uses largest component
            - May not converge for bipartite graphs
        """
        try:
            logger.debug("Starting eigenvector centrality calculation")

            nodes = self._get_node_list(node_list)

            # For disconnected graphs, use largest component
            if not nx.is_connected(self.graph):
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc).copy()
                logger.warning(
                    f"Disconnected graph detected. Using largest component "
                    f"({len(largest_cc)} nodes)"
                )
            else:
                subgraph = self.graph

            # Calculate eigenvector centrality
            centrality = nx.eigenvector_centrality(
                subgraph,
                max_iter=max_iter,
                tol=tol
            )

            # For disconnected graph, add isolated nodes
            if not nx.is_connected(self.graph):
                for node in self.graph.nodes():
                    if node not in centrality:
                        centrality[node] = 0.0

            # Filter if needed
            if node_list is not None:
                centrality = {n: centrality[n] for n in nodes if n in centrality}

            self._validate_centrality_output(centrality, (0.0, 1.0))

            logger.info(
                f"Eigenvector centrality calculated: {len(centrality)} nodes"
            )

            return centrality

        except Exception as e:
            logger.error(f"Eigenvector centrality calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate eigenvector centrality: {str(e)}")

    # ========================================================================
    # METHOD 1.4: CLOSENESS CENTRALITY (Preview for Phase 1)
    # ========================================================================

    def closeness_centrality(
        self,
        node_list: Optional[List[str]] = None,
        distance: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate closeness centrality for all or specified nodes.

        Closeness centrality measures how quickly a node can reach others.
        A node with high closeness is close to all others on average.

        For node v:

            CC(v) = (n-1) / Σ(d(v, u) for all u != v)

        where d(v, u) is shortest path distance.

        Range: [0, 1]

        Args:
            node_list: Specific nodes to calculate
            distance: Edge weight attribute (for weighted graphs)

        Returns:
            Dict mapping node_id -> centrality_value

        Raises:
            RuntimeError: If calculation fails

        Use Cases:
            - Identify actors who quickly spread information
            - Find central hubs
            - Information dissemination analysis

        Note:
            - For disconnected graphs, uses harmonic mean
        """
        try:
            logger.debug("Starting closeness centrality calculation")

            nodes = self._get_node_list(node_list)

            # Use harmonic closeness for robustness
            centrality = nx.closeness_centrality(
                self.graph,
                distance=distance
            )

            # Filter if needed
            if node_list is not None:
                centrality = {n: centrality[n] for n in nodes if n in centrality}

            logger.info(
                f"Closeness centrality calculated: {len(centrality)} nodes"
            )

            return centrality

        except Exception as e:
            logger.error(f"Closeness centrality calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate closeness centrality: {str(e)}")

    # ========================================================================
    # METHOD 1.5: CLUSTERING COEFFICIENT (Preview for Phase 1)
    # ========================================================================

    def clustering_coefficient(
        self,
        node_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate clustering coefficient for all or specified nodes.

        Clustering coefficient measures local clustering around each node.
        Indicates how tightly knit a node's neighborhood is.

        For node v with k neighbors:

            CC(v) = 2*t(v) / (k(k-1))

        where t(v) = number of triangles through v.

        Range: [0, 1]
            - 0: No triangles involving node
            - 1: Node's neighbors form complete clique

        Args:
            node_list: Specific nodes to calculate

        Returns:
            Dict mapping node_id -> clustering_coefficient

        Raises:
            RuntimeError: If calculation fails

        Use Cases:
            - Identify tight-knit groups
            - Detect friend circles or close alliances
            - Measure local network cohesion
        """
        try:
            logger.debug("Starting clustering coefficient calculation")

            nodes = self._get_node_list(node_list)

            # Calculate clustering coefficient
            clustering = nx.clustering(self.graph)

            # Filter if needed
            if node_list is not None:
                clustering = {n: clustering[n] for n in nodes if n in clustering}

            self._validate_centrality_output(clustering, (0.0, 1.0))

            logger.info(
                f"Clustering coefficient calculated: {len(clustering)} nodes"
            )

            return clustering

        except Exception as e:
            logger.error(f"Clustering coefficient calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate clustering coefficient: {str(e)}")

    # ========================================================================
    # GRAPH-LEVEL METRICS (Method 1.6)
    # ========================================================================

    def calculate_graph_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Calculate comprehensive graph-level metrics.

        Computes density, average clustering, diameter, and other
        graph-wide summary statistics.

        Returns:
            Dictionary with keys:
            - density (float): Network density [0, 1]
            - avg_clustering (float): Average clustering coefficient
            - avg_path_length (float): Average shortest path length
            - diameter (int): Longest shortest path
            - num_components (int): Number of connected components
            - num_nodes (int): Total nodes
            - num_edges (int): Total edges
            - avg_degree (float): Average node degree
            - max_degree (int): Maximum node degree
            - min_degree (int): Minimum node degree
            - is_connected (bool): True if graph is connected
            - timestamp (datetime): Calculation timestamp

        Example:
            >>> G = nx.complete_graph(5, create_using=nx.Graph)
            >>> calc = DescriptiveMetricsCalculator(G)
            >>> metrics = calc.calculate_graph_metrics()
            >>> print(metrics['density'])
            1.0
        """
        try:
            logger.debug("Starting graph-level metrics calculation")

            # Basic metrics
            density = nx.density(self.graph)

            # Average clustering coefficient
            avg_clustering = nx.average_clustering(self.graph)

            # Degree statistics
            degrees = [degree for node, degree in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            max_degree = np.max(degrees) if degrees else 0
            min_degree = np.min(degrees) if degrees else 0

            # Connected components
            num_components = nx.number_connected_components(self.graph)
            is_connected = nx.is_connected(self.graph)

            # Path length and diameter (for connected graphs)
            if is_connected:
                avg_path_length = nx.average_shortest_path_length(self.graph)
                diameter = nx.diameter(self.graph)
            else:
                # For disconnected graphs, calculate per component
                path_lengths = []
                diameters = []
                for component in nx.connected_components(self.graph):
                    subgraph = self.graph.subgraph(component)
                    if len(subgraph) > 1:
                        path_lengths.append(
                            nx.average_shortest_path_length(subgraph)
                        )
                        diameters.append(nx.diameter(subgraph))

                avg_path_length = np.mean(path_lengths) if path_lengths else 0
                diameter = max(diameters) if diameters else 0

            metrics = {
                'density': float(density),
                'avg_clustering': float(avg_clustering),
                'avg_path_length': float(avg_path_length),
                'diameter': int(diameter),
                'num_components': num_components,
                'num_nodes': self.node_count,
                'num_edges': self.edge_count,
                'avg_degree': float(avg_degree),
                'max_degree': int(max_degree),
                'min_degree': int(min_degree),
                'is_connected': is_connected,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Graph metrics calculated: density={density:.4f}, "
                f"components={num_components}, diameter={diameter}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Graph metrics calculation failed: {str(e)}")
            raise RuntimeError(f"Failed to calculate graph metrics: {str(e)}")


# ============================================================================
# DATABASE INTEGRATION FUNCTIONS
# ============================================================================

def create_metrics_tables(db_connection: sqlite3.Connection) -> None:
    """
    Create required metrics tables in database if not exist.

    Args:
        db_connection: SQLite connection object

    Raises:
        sqlite3.Error: If table creation fails

    Tables Created:
        - network_metrics_nodes: Node-level metrics
        - network_metrics_graph: Graph-level metrics
    """
    try:
        cursor = db_connection.cursor()

        # Node-level metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_metrics_nodes (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                metric_date DATE NOT NULL,

                degree_centrality REAL CHECK(degree_centrality >= 0 AND degree_centrality <= 1),
                betweenness_centrality REAL CHECK(betweenness_centrality >= 0 AND betweenness_centrality <= 1),
                eigenvector_centrality REAL CHECK(eigenvector_centrality >= 0 AND eigenvector_centrality <= 1),
                closeness_centrality REAL CHECK(closeness_centrality >= 0 AND closeness_centrality <= 1),
                clustering_coefficient REAL CHECK(clustering_coefficient >= 0 AND clustering_coefficient <= 1),

                network_filter TEXT DEFAULT 'all',
                calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                method_version TEXT DEFAULT '1.0',

                UNIQUE(actor_id, metric_date, network_filter)
            )
        ''')

        # Graph-level metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_metrics_graph (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE NOT NULL,

                num_nodes INTEGER NOT NULL,
                num_edges INTEGER NOT NULL,
                density REAL CHECK(density >= 0 AND density <= 1),
                avg_clustering_coefficient REAL CHECK(avg_clustering_coefficient >= 0 AND avg_clustering_coefficient <= 1),
                avg_path_length REAL,
                diameter INTEGER,
                num_connected_components INTEGER,
                avg_degree REAL,
                max_degree INTEGER,
                min_degree INTEGER,
                is_connected BOOLEAN,

                network_filter TEXT DEFAULT 'all',
                calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                method_version TEXT DEFAULT '1.0',

                UNIQUE(metric_date, network_filter)
            )
        ''')

        # Create indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_nodes_actor 
            ON network_metrics_nodes(actor_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_nodes_date 
            ON network_metrics_nodes(metric_date)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_graph_date 
            ON network_metrics_graph(metric_date)
        ''')

        db_connection.commit()
        logger.info("Metrics tables created successfully")

    except sqlite3.Error as e:
        logger.error(f"Error creating metrics tables: {str(e)}")
        raise


def save_node_metrics(
    db_connection: sqlite3.Connection,
    actor_id: str,
    metric_date: str,
    metrics: Dict[str, float],
    network_filter: str = 'all',
    method_version: str = '1.0'
) -> int:
    """
    Save node-level metrics to database.

    Args:
        db_connection: SQLite connection
        actor_id: Actor identifier
        metric_date: Calculation date (YYYY-MM-DD)
        metrics: Dictionary with centrality metrics
        network_filter: Network scope identifier
        method_version: Algorithm version

    Returns:
        Inserted metric_id

    Raises:
        sqlite3.Error: If insert fails
    """
    try:
        cursor = db_connection.cursor()

        cursor.execute('''
            INSERT INTO network_metrics_nodes 
            (actor_id, metric_date, degree_centrality, betweenness_centrality,
             eigenvector_centrality, closeness_centrality, clustering_coefficient,
             network_filter, method_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(actor_id, metric_date, network_filter)
            DO UPDATE SET 
                degree_centrality = excluded.degree_centrality,
                betweenness_centrality = excluded.betweenness_centrality,
                eigenvector_centrality = excluded.eigenvector_centrality,
                closeness_centrality = excluded.closeness_centrality,
                clustering_coefficient = excluded.clustering_coefficient,
                calculation_timestamp = CURRENT_TIMESTAMP,
                method_version = excluded.method_version
        ''', (
            actor_id,
            metric_date,
            metrics.get('degree_centrality'),
            metrics.get('betweenness_centrality'),
            metrics.get('eigenvector_centrality'),
            metrics.get('closeness_centrality'),
            metrics.get('clustering_coefficient'),
            network_filter,
            method_version
        ))

        db_connection.commit()
        return cursor.lastrowid

    except sqlite3.Error as e:
        logger.error(f"Error saving node metrics: {str(e)}")
        raise


def save_graph_metrics(
    db_connection: sqlite3.Connection,
    metric_date: str,
    metrics: Dict[str, Union[float, int]],
    network_filter: str = 'all',
    method_version: str = '1.0'
) -> int:
    """
    Save graph-level metrics to database.

    Args:
        db_connection: SQLite connection
        metric_date: Calculation date (YYYY-MM-DD)
        metrics: Dictionary with graph metrics
        network_filter: Network scope identifier
        method_version: Algorithm version

    Returns:
        Inserted metric_id

    Raises:
        sqlite3.Error: If insert fails
    """
    try:
        cursor = db_connection.cursor()

        cursor.execute('''
            INSERT INTO network_metrics_graph
            (metric_date, num_nodes, num_edges, density, avg_clustering_coefficient,
             avg_path_length, diameter, num_connected_components, avg_degree,
             max_degree, min_degree, is_connected, network_filter, method_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(metric_date, network_filter)
            DO UPDATE SET
                num_nodes = excluded.num_nodes,
                num_edges = excluded.num_edges,
                density = excluded.density,
                avg_clustering_coefficient = excluded.avg_clustering_coefficient,
                avg_path_length = excluded.avg_path_length,
                diameter = excluded.diameter,
                num_connected_components = excluded.num_connected_components,
                avg_degree = excluded.avg_degree,
                max_degree = excluded.max_degree,
                min_degree = excluded.min_degree,
                is_connected = excluded.is_connected,
                calculation_timestamp = CURRENT_TIMESTAMP,
                method_version = excluded.method_version
        ''', (
            metric_date,
            metrics['num_nodes'],
            metrics['num_edges'],
            metrics['density'],
            metrics['avg_clustering'],
            metrics['avg_path_length'],
            metrics['diameter'],
            metrics['num_components'],
            metrics['avg_degree'],
            metrics['max_degree'],
            metrics['min_degree'],
            metrics['is_connected'],
            network_filter,
            method_version
        ))

        db_connection.commit()
        return cursor.lastrowid

    except sqlite3.Error as e:
        logger.error(f"Error saving graph metrics: {str(e)}")
        raise
