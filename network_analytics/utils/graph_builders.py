#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import networkx as nx


def build_actor_network(db_path, relationship_types=None):
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT source_actor_id, target_actor_id FROM edges WHERE 1=1'
        params = []
        
        if relationship_types:
            placeholders = ','.join(['?' for _ in relationship_types])
            query += f' AND relationship_type IN ({placeholders})'
            params.extend(relationship_types)
        
        cursor.execute(query, params)
        edges = cursor.fetchall()
        conn.close()
        
        # Create graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        print(f"Loaded network: {len(G)} nodes, {G.number_of_edges()} edges")
        
        return G
    
    except Exception as e:
        print(f"Error building actor network: {str(e)}")
        raise
