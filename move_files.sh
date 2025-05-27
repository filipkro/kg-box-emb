#!/bin/bash

mv large_files/all_graphs-no-int.ttl graphs/
mv large_files/all_graphs.ttl graphs/
mv large_files/full-kg-no-int.ttl graphs/
mv large_files/full-kg.ttl graphs/
mv large_files/chebi.ttl graphs/
mv large_files/go-ext.ttl graphs/
mv large_files/kg-nf-no-int.ttl graphs/
mv large_files/kg-nf.ttl graphs/

mv large_files/interactions_DMA30.pkl datasets/split_datasets/
mv large_files/interactions.pkl datasets/split_datasets/
mv large_files/refined_interactions_DMA30.pkl datasets/split_datasets/
mv large_files/refined_interactions.pkl datasets/split_datasets/
mv large_files/pyg_graph_c_DMA30_fitness.pkl datasets/split_datasets/

mv large_files/prel_explanationsDMA30-InputXGradient-full_model-feb02-10000-cleaned.tsv explanations/
mv large_files/prel_explanationsDMA30-InputXGradient-full_model-feb02-10000.tsv explanations/

mv large_files/aao1729_data_s1.tsv data/interaction_data/
mv large_files/SGA_DAmP.txt data/interaction_data/
mv large_files/SGA_ExE.txt data/interaction_data/
mv large_files/SGA_ExN_NxE.txt data/interaction_data/
mv large_files/SGA_NxN.txt data/interaction_data/

mv large_files/sgd-data-ext.pkl data/
mv large_files/sgd-data-slim.pkl data/
