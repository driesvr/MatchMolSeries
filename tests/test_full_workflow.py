from matchmolseries import MatchMolSeries
import pandas as pd, numpy as np
import time
import polars as pl
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



n_compounds_to_load = 100000
n_compounds_to_process = 35000
n_query_compounds = 500
min_series_length = 5
total_start_time = time.time()

mms = MatchMolSeries()
logger.info("\n=== Loading and Processing Input Data ===")
data_start = time.time()
# Example operations
dataset = pd.read_csv('BindingDB_Patents.tsv', sep='\t', on_bad_lines='skip', nrows=n_compounds_to_load)
dataset['smiles'] = dataset['Ligand SMILES']
dataset['potency'] = np.round(9 - np.log10(pd.to_numeric(dataset['IC50 (nM)'], errors='coerce')),2)

dataset.dropna(subset=['smiles', 'potency'], inplace=True)
logger.info(f'{len(dataset)} molecules loaded, will use first {n_compounds_to_process} for creating database and last {n_query_compounds} for querying')
dataset.reset_index(drop=True, inplace=True)

input_df = dataset[:n_compounds_to_process].copy()
logger.info(f'Data loading completed in {time.time() - data_start:.2f} seconds')
logger.info(f'{len(input_df)} molecules to fragment')
logger.info(input_df['BindingDB Entry DOI'].value_counts())

query_df = dataset[n_compounds_to_process:n_compounds_to_process+n_query_compounds].copy()
logger.info(f'starting query at index: {n_compounds_to_process}, ending at index: {n_compounds_to_process+n_query_compounds}')
logger.info(f'{len(query_df)} query molecules')

logger.info("\n=== Fragmenting Molecules ===")
frag_start = time.time()

# Try to load reference fragments from parquet if it exists
parquet_path = "reference_fragments.parquet"
try:
    mms.load_fragments(parquet_path)
    logger.info(f"Loaded reference fragments from {parquet_path}")
except:
    logger.info("No existing reference fragments found. Processing molecules...")
    mms.fragment_molecules(input_df, assay_col='BindingDB Entry DOI')
    mms.fragments_df.write_parquet(parquet_path)
    logger.info(f"Saved reference fragments to {parquet_path}")

logger.info(f'Fragmentation completed in {time.time() - frag_start:.2f} seconds')

logger.info("\n=== Querying Fragments ===")
query_start = time.time()
result = mms.query_fragments(query_df, min_series_length=min_series_length, assay_col='BindingDB Entry DOI')
logger.info(f'Found {len(result)} matches')
logger.info(f'Query completed in {time.time() - query_start:.2f} seconds')

logger.info("\n=== Putting compounds back together ===")
combine_start = time.time()
result = mms.combine_fragments(result)
logger.info(f'Combination completed in {time.time() - combine_start:.2f} seconds')

logger.info(result[:10])
result[:10].to_csv('test_res.csv')

logger.info(f"\nTotal execution time: {time.time() - total_start_time:.2f} seconds")
