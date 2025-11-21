#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sqlite3
# from datetime import date
# from tkinter import messagebox
# from network_analytics.descriptive_metrics import save_node_metrics
# from network_analytics.utils.graph_builders import build_actor_network
# from network_analytics.descriptive_metrics import DescriptiveMetricsCalculator


# def on_save_metrics_to_database(self):
#     try:        
#         # Calculate metrics if not already done
#         if not hasattr(self, 'degree_centrality_results'):
#             G = build_actor_network('network_analytics.db')
#             calc = DescriptiveMetricsCalculator(G)
#             self.degree_centrality_results = calc.degree_centrality()
        
#         # Connect to database
#         conn = sqlite3.connect('network_analytics.db')
        
#         # Save each actor's metrics
#         for actor_id, dc_value in self.degree_centrality_results.items():
            
#             # Prepare metrics dictionary
#             metrics_to_save = {
#                 'degree_centrality': dc_value,
#                 'betweenness_centrality': None,
#                 'eigenvector_centrality': None,
#                 'closeness_centrality': None,
#                 'clustering_coefficient': None
#             }
            
#             # Save to database
#             metric_id = save_node_metrics(
#                 db_connection=conn,
#                 actor_id=actor_id,
#                 metric_date=str(date.today()),
#                 metrics=metrics_to_save,
#                 network_filter='all',
#                 method_version='1.0'
#             )
            
#             print(f"Saved metric with ID: {metric_id}")
        
#         conn.close()
#         self.results_text.insert(tk.END, 
#             f"\nMetrics saved to database on {date.today()}")
#         messagebox.showinfo("Success", "Metrics saved!")
    
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to save: {str(e)}")
