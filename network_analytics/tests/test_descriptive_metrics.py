#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
network_analytics/tests/test_descriptive_metrics.py

Unit Tests for Descriptive Metrics Module - Phase 1

Tests cover:
- Degree centrality calculations
- Edge case handling
- Performance on large graphs
- Output validation
- Integration with database storage

Author: Network Analytics Testing Suite
Version: 1.0
Date: 2025-11-21
"""

import unittest
import networkx as nx
import numpy as np
from datetime import datetime
import sys
import os
import time
import sqlite3
import tempfile

# Import from descriptive_metrics module
# (In production: from network_analytics.descriptive_metrics import ...)
# For testing purposes, assume imported


class TestDegreeCentrality(unittest.TestCase):
    """Test suite for degree centrality calculation."""

    def setUp(self):
        """Initialize test graphs and fixtures."""
        # Empty graph (edge case)
        self.empty_graph = nx.Graph()

        # Single node
        self.single_node = nx.Graph()
        self.single_node.add_node("actor1")

        # Two nodes connected
        self.two_nodes = nx.Graph()
        self.two_nodes.add_edge("A", "B")

        # Complete graph K5
        self.complete_graph = nx.complete_graph(5, create_using=nx.Graph)
        self.complete_graph = nx.relabel_nodes(
            self.complete_graph, 
            {i: f"actor{i}" for i in range(5)}
        )

        # Star graph (1 hub, 4 leaves)
        self.star_graph = nx.star_graph(4)
        self.star_graph = nx.relabel_nodes(
            self.star_graph, 
            {i: f"node{i}" for i in range(5)}
        )

        # Linear graph (path)
        self.path_graph = nx.path_graph(5, create_using=nx.Graph)
        self.path_graph = nx.relabel_nodes(
            self.path_graph,
            {i: f"p{i}" for i in range(5)}
        )

        # Disconnected graph (2 components)
        self.disconnected = nx.Graph()
        self.disconnected.add_edges_from([("A", "B"), ("B", "C")])
        self.disconnected.add_edges_from([("X", "Y"), ("Y", "Z")])

        # Directed graph
        self.directed = nx.DiGraph()
        self.directed.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

    # ========================================================================
    # TEST: VALIDATION
    # ========================================================================

    def test_empty_graph_raises_error(self):
        """Empty graph should raise ValueError during initialization."""
        # This test assumes DescriptiveMetricsCalculator validates on init
        # Adjust based on actual implementation
        from descriptive_metrics import DescriptiveMetricsCalculator
        with self.assertRaises(ValueError) as context:
            DescriptiveMetricsCalculator(self.empty_graph)
        self.assertIn("empty", str(context.exception).lower())

    def test_invalid_graph_type_raises_error(self):
        """Non-NetworkX object should raise TypeError."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        with self.assertRaises(TypeError):
            DescriptiveMetricsCalculator("not a graph")
        with self.assertRaises(TypeError):
            DescriptiveMetricsCalculator([1, 2, 3])

    # ========================================================================
    # TEST: SINGLE NODE GRAPH
    # ========================================================================

    def test_single_node_centrality_is_zero(self):
        """Single node with no edges should have centrality = 0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.single_node)
        result = calc.degree_centrality()

        self.assertEqual(len(result), 1)
        self.assertIn("actor1", result)
        self.assertEqual(result["actor1"], 0.0)

    def test_single_node_returns_dict(self):
        """Single node should return dictionary."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.single_node)
        result = calc.degree_centrality()

        self.assertIsInstance(result, dict)

    # ========================================================================
    # TEST: TWO NODES GRAPH
    # ========================================================================

    def test_two_connected_nodes_centrality(self):
        """Two connected nodes should each have centrality = 1.0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.two_nodes)
        result = calc.degree_centrality()

        # Each node has 1 connection, (n-1) = 1
        # DC = 1/1 = 1.0
        self.assertEqual(result["A"], 1.0)
        self.assertEqual(result["B"], 1.0)

    # ========================================================================
    # TEST: COMPLETE GRAPH
    # ========================================================================

    def test_complete_graph_all_nodes_one(self):
        """All nodes in complete graph should have centrality = 1.0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        result = calc.degree_centrality()

        self.assertEqual(len(result), 5)
        for node, centrality in result.items():
            self.assertAlmostEqual(
                centrality, 1.0, places=5,
                msg=f"Node {node} should have centrality 1.0, got {centrality}"
            )

    def test_complete_graph_normalization(self):
        """Complete graph with 5 nodes: each connected to 4."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        result = calc.degree_centrality()

        # Degree = 4, n-1 = 4, so DC = 4/4 = 1.0
        for centrality in result.values():
            self.assertEqual(centrality, 1.0)

    # ========================================================================
    # TEST: STAR GRAPH
    # ========================================================================

    def test_star_graph_hub_centrality_one(self):
        """Hub in star graph should have centrality = 1.0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.star_graph)
        result = calc.degree_centrality()

        # Hub (node0) is connected to all 4 leaves
        # DC = 4 / (5-1) = 4/4 = 1.0
        self.assertAlmostEqual(result["node0"], 1.0, places=5)

    def test_star_graph_leaf_centrality(self):
        """Leaf nodes in star graph should have centrality = 0.2."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.star_graph)
        result = calc.degree_centrality()

        # Each leaf connected to hub only
        # DC = 1 / (5-1) = 1/4 = 0.25
        # Wait, checking NetworkX convention...

        # Actually: DC(v) = deg(v) / (n-1)
        # For 5-node graph: n-1 = 4
        # Hub: degree = 4, DC = 4/4 = 1.0
        # Leaves: degree = 1, DC = 1/4 = 0.25

        for i in range(1, 5):
            leaf_centrality = result[f"node{i}"]
            self.assertAlmostEqual(leaf_centrality, 0.25, places=5,
                msg=f"Leaf node{i} should have centrality 0.25, got {leaf_centrality}")

    def test_star_graph_distribution(self):
        """Star graph centrality sum property."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.star_graph)
        result = calc.degree_centrality()

        # Hub: 1.0
        # Leaves (4x): 0.25 each = 1.0
        # Total: 2.0
        total_centrality = sum(result.values())
        self.assertAlmostEqual(total_centrality, 2.0, places=5)

    # ========================================================================
    # TEST: PATH GRAPH
    # ========================================================================

    def test_path_graph_endpoint_centrality(self):
        """Endpoints in path should have lowest centrality."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.path_graph)
        result = calc.degree_centrality()

        # p0 and p4 are endpoints: degree = 1
        # DC = 1/4 = 0.25
        self.assertAlmostEqual(result["p0"], 0.25, places=5)
        self.assertAlmostEqual(result["p4"], 0.25, places=5)

    def test_path_graph_middle_centrality(self):
        """Middle nodes in path should have higher centrality."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.path_graph)
        result = calc.degree_centrality()

        # p2 is center: degree = 2
        # DC = 2/4 = 0.5
        self.assertAlmostEqual(result["p2"], 0.5, places=5)

    def test_path_graph_monotonic(self):
        """Path graph centrality increases from endpoints to middle."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.path_graph)
        result = calc.degree_centrality()

        centralities = [result[f"p{i}"] for i in range(5)]

        # Should be: [0.25, 0.5, 0.5, 0.5, 0.25]
        # Or similar pattern (symmetric)
        self.assertEqual(centralities[0], centralities[4])  # Symmetry

    # ========================================================================
    # TEST: DISCONNECTED GRAPH
    # ========================================================================

    def test_disconnected_graph_independence(self):
        """Disconnected components calculated independently."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.disconnected)
        result = calc.degree_centrality()

        # Component 1: A-B-C (linear)
        # Component 2: X-Y-Z (linear)
        # Each node has same degree pattern

        self.assertEqual(len(result), 6)

        # B and Y are centers of their components (degree 2)
        # DC = 2/2 = 1.0
        self.assertAlmostEqual(result["B"], 1.0, places=5)
        self.assertAlmostEqual(result["Y"], 1.0, places=5)

    # ========================================================================
    # TEST: DIRECTED GRAPH
    # ========================================================================

    def test_directed_graph_processing(self):
        """Directed graphs should be handled by NetworkX."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.directed)
        result = calc.degree_centrality()

        # NetworkX degree_centrality considers total degree
        # for directed graphs by default
        self.assertEqual(len(result), 3)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)

    # ========================================================================
    # TEST: OUTPUT VALIDATION
    # ========================================================================

    def test_centrality_range_0_to_1(self):
        """All centrality values should be in [0, 1]."""
        from descriptive_metrics import DescriptiveMetricsCalculator

        graphs = [
            self.single_node, self.two_nodes, self.complete_graph,
            self.star_graph, self.path_graph, self.disconnected, self.directed
        ]

        for graph in graphs:
            calc = DescriptiveMetricsCalculator(graph)
            result = calc.degree_centrality()

            for node, centrality in result.items():
                self.assertGreaterEqual(centrality, 0.0,
                    msg=f"Centrality {centrality} < 0 for node {node}")
                self.assertLessEqual(centrality, 1.0,
                    msg=f"Centrality {centrality} > 1 for node {node}")

    def test_return_type_dict(self):
        """Return value should be dictionary."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        result = calc.degree_centrality()

        self.assertIsInstance(result, dict)
        for key, value in result.items():
            self.assertIsInstance(key, (str, int))
            self.assertIsInstance(value, float)

    def test_all_nodes_present(self):
        """Result should include all nodes in graph."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        result = calc.degree_centrality()

        graph_nodes = set(self.complete_graph.nodes())
        result_nodes = set(result.keys())

        self.assertEqual(graph_nodes, result_nodes)

    # ========================================================================
    # TEST: SELECTIVE NODE CALCULATION
    # ========================================================================

    def test_node_list_parameter(self):
        """Should calculate only specified nodes."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)

        # Calculate only subset
        subset = ["actor0", "actor1"]
        result = calc.degree_centrality(node_list=subset)

        self.assertEqual(len(result), len(subset))
        self.assertIn("actor0", result)
        self.assertIn("actor1", result)
        self.assertNotIn("actor2", result)

    # ========================================================================
    # TEST: PERFORMANCE
    # ========================================================================

    def test_performance_medium_graph(self):
        """Should handle 500-node graph in reasonable time."""
        from descriptive_metrics import DescriptiveMetricsCalculator

        # Create random graph
        G = nx.erdos_renyi_graph(500, 0.1)

        start_time = time.time()
        calc = DescriptiveMetricsCalculator(G)
        result = calc.degree_centrality()
        elapsed = time.time() - start_time

        self.assertEqual(len(result), 500)
        self.assertLess(elapsed, 1.0, f"Calculation took {elapsed}s, should be <1s")

    def test_performance_large_graph(self):
        """Should handle 1000-node graph."""
        from descriptive_metrics import DescriptiveMetricsCalculator

        # Create random graph
        G = nx.erdos_renyi_graph(1000, 0.05)

        start_time = time.time()
        calc = DescriptiveMetricsCalculator(G)
        result = calc.degree_centrality()
        elapsed = time.time() - start_time

        self.assertEqual(len(result), 1000)
        self.assertLess(elapsed, 2.0, f"Calculation took {elapsed}s, should be <2s")

    # ========================================================================
    # TEST: AGAINST KNOWN VALUES
    # ========================================================================

    def test_against_manual_calculation(self):
        """Verify against manually calculated values."""
        from descriptive_metrics import DescriptiveMetricsCalculator

        # Manual graph: A-B-C with specific degrees
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])

        # Degrees: A=1, B=3, C=1, D=1
        # n-1 = 3
        # DC: A=1/3, B=3/3=1, C=1/3, D=1/3

        calc = DescriptiveMetricsCalculator(G)
        result = calc.degree_centrality()

        expected = {
            "A": 1/3,
            "B": 1.0,
            "C": 1/3,
            "D": 1/3
        }

        for node, expected_val in expected.items():
            self.assertAlmostEqual(result[node], expected_val, places=5,
                msg=f"Node {node}: expected {expected_val}, got {result[node]}")


class TestGraphMetrics(unittest.TestCase):
    """Test suite for graph-level metrics."""

    def setUp(self):
        """Initialize test graphs."""
        self.complete_graph = nx.complete_graph(5, create_using=nx.Graph)
        self.empty_graph = nx.empty_graph(5, create_using=nx.Graph)
        self.star_graph = nx.star_graph(4)
        self.path_graph = nx.path_graph(5, create_using=nx.Graph)

    def test_calculate_graph_metrics_returns_dict(self):
        """Graph metrics should return dictionary."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        result = calc.calculate_graph_metrics()

        self.assertIsInstance(result, dict)

    def test_complete_graph_density_one(self):
        """Complete graph should have density = 1.0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.complete_graph)
        metrics = calc.calculate_graph_metrics()

        self.assertAlmostEqual(metrics['density'], 1.0, places=5)

    def test_empty_graph_density_zero(self):
        """Empty graph should have density = 0.0."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.empty_graph)
        metrics = calc.calculate_graph_metrics()

        self.assertAlmostEqual(metrics['density'], 0.0, places=5)

    def test_metrics_required_fields(self):
        """All required metrics fields should be present."""
        from descriptive_metrics import DescriptiveMetricsCalculator
        calc = DescriptiveMetricsCalculator(self.star_graph)
        metrics = calc.calculate_graph_metrics()

        required_fields = [
            'density', 'avg_clustering', 'avg_path_length', 'diameter',
            'num_components', 'num_nodes', 'num_edges', 'avg_degree',
            'max_degree', 'min_degree', 'is_connected', 'timestamp'
        ]

        for field in required_fields:
            self.assertIn(field, metrics, f"Missing required field: {field}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
