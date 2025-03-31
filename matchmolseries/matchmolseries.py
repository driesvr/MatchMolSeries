"""Core implementation of the MatchMolSeries class for molecular fragmentation and analysis."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import polars as pl
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatchMolSeries:
    """
    MatchMolSeries (Matched Molecular Series) class for molecular fragmentation and analysis.
    
    A class that provides comprehensive functionality for analyzing and processing molecular series
    through fragmentation, organization, and querying of chemical structures.
    
    Key Features
    -----------
    * Molecular fragmentation using SMARTS-based bond breaking transformations
    * Organization of fragments by core structure with attachment point tracking
    * Fragment database querying and analysis
    * Efficient data storage using Polars DataFrames
    
    Examples
    --------
    >>> mms = MatchMolSeries()
    >>> df = pd.DataFrame({'smiles': ['c1ccccc1F'], 'potency': [1.0], 'assay': ['assay1']})
    >>> fragments = mms.fragment_molecules(df)
    """

    def __init__(self):
        """
        Initialize MatchMolSeries
        """
        # Pre-compile the reactions
        self.combine_reaction = AllChem.ReactionFromSmarts('[*:1][At].[At][*:2]>>[*:1]-[*:2]')
        self.splitting_reactions = [
            AllChem.ReactionFromSmarts('[*;R:1]-!@[*:2]>>[*:1][At].[At][*:2]'),
            AllChem.ReactionFromSmarts('[!#6;!R:1]-!@[C;!X3;!R:2]>>[*:1][At].[At][*:2]'),
            AllChem.ReactionFromSmarts('[*!H0:1]>>[*:1][At].[At]')

        ]
        self.attachment_point_substruct = Chem.MolFromSmarts('[*:1][At]')
    
    def load_fragments(self, fragments_file: str):
        """
        Load fragment data from a Parquet or csv file.
        
        Parameters
        ----------
        fragments_file : str
            Path to the Parquet or csv file containing fragment data.
            Path must end in .parquet or .csv
        """
        if fragments_file.endswith('.parquet'):
            self.fragments_df = pl.read_parquet(fragments_file)
        elif fragments_file.endswith('.csv'):
            self.fragments_df = pl.read_csv(fragments_file)
        else:
            raise ValueError(f"Unsupported file format: {fragments_file}")
    
    def save_fragments(self, fragments_file_path: str):
        """
        Write fragment data to a Parquet or csv file.
        
        Parameters
        ----------
        fragments_file_path : str
            Path to the Parquet or csv file to write fragment data to.
        """
        if fragments_file_path.endswith('.parquet'):
            self.fragments_df.write_parquet(fragments_file_path)
        elif fragments_file_path.endswith('.csv'):
            self.fragments_df.write_csv(fragments_file_path)
        else:
            raise ValueError(f"Unsupported file format: {fragments_file_path}")
    
    def _standardize_mol(self, mol: Chem.Mol, frag_remover: rdMolStandardize.FragmentRemover) -> Chem.Mol:
        """
        Standardize a molecule using RDKit standardization methods.
        
        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule to standardize.
        frag_remover : rdMolStandardize.FragmentRemover
            RDKit fragment remover instance for removing salt fragments
        
        Returns
        -------
        Chem.Mol
            Standardized RDKit molecule with hydrogens removed and sanitized structure.
        """
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        mol = rdMolStandardize.Normalize(mol)
        mol = rdMolStandardize.Reionize(mol)

        mol = frag_remover.remove(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol



    def fragment_molecules(self, input_df: pd.DataFrame, query_or_ref: str = 'ref',
                     smiles_col: str = 'smiles', potency_col: str = 'potency',
                     assay_col: str = 'assay', max_mol_atomcount: float = 100,
                     standardize: bool = True, 
                     max_fragsize_fraction: float = 0.5) -> pl.DataFrame:
        """
        Fragment molecules from an input DataFrame using SMARTS-based chemical transformations.
        
        Parameters
        ----------
        input_df : pandas.DataFrame
            Input DataFrame containing molecule information
        query_or_ref : {'ref', 'query'}, default='ref'
            Specify whether this is a reference or query set. Results will be stored in
            self.fragments_df for 'ref' or self.query_fragments_df for 'query'
        smiles_col : str, default='smiles'
            Name of column containing SMILES strings
        potency_col : str, default='potency'
            Name of column containing potency values
        assay_col : str, default='assay'
            Name of column containing assay identifiers
        max_mol_atomcount : float, default=100
            Maximum number of heavy atoms allowed in a molecule
        standardize : bool, default=True
            Whether to standardize molecules using RDKit's StandardizeMol
        max_fragsize_fraction : float, default=0.5
            Maximum allowed size of fragment relative to parent molecule (0.0-1.0)
            
        Returns
        -------
        polars.DataFrame
            DataFrame containing fragment information. 
            DataFrame is additionally stored in self.fragments_df for reference sets
            or self.query_fragments_df for query sets
            
        Raises
        ------
        ValueError
            If input DataFrame is empty or missing required columns
        """
        if input_df.empty:
            raise ValueError("Input DataFrame is empty")
        if not all(col in input_df.columns for col in [smiles_col, potency_col, assay_col]):
            raise ValueError(f"Missing required columns: {smiles_col}, {potency_col}, or {assay_col}")

        frag_remover = rdMolStandardize.FragmentRemover()

        # Initialize lists to store fragment information
        fragment_smiles_list = []
        molecule_indices = []
        cut_indices = []
        fragment_potencies = []
        fragment_sizes = []
        parent_smiles_list = []
        core_smiles_list = []
        assay_list = []
        
        # Process each molecule
        input_df = input_df.sort_values(by=assay_col, ascending=False)
        for mol_idx, (smiles, potency, assay) in enumerate(input_df[[smiles_col, potency_col, assay_col]].values):
            if (mol_idx % 1000 == 0) & (mol_idx>0):
                logger.info(f"Processed {mol_idx} molecules...")
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                logger.warning(f'Molecule {mol_idx} is None')
                continue
            
            if standardize:
                try:
                    mol = self._standardize_mol(mol, frag_remover)
                except Exception as e:
                    logger.error(f'Failed to standardize molecule {mol_idx}: {e}')
                    continue
            
            parent_atom_count = mol.GetNumHeavyAtoms()
            if parent_atom_count > max_mol_atomcount:
                continue
            
            # Apply transformations
            products = []
            for rxn in self.splitting_reactions:
                products.extend(rxn.RunReactants((mol,)))
            
            # Process each product
            for cut_idx, frags in enumerate(products):
                if not frags or len(frags) != 2:
                    continue
                    
                try:
                    [Chem.SanitizeMol(frag) for frag in frags]
                except:
                    continue
                
                # Get sizes and determine core/fragment
                frag1_size = frags[0].GetNumHeavyAtoms() - 1
                frag2_size = frags[1].GetNumHeavyAtoms() - 1
                
                if frag1_size >= frag2_size:
                    core, fragment = frags[0], frags[1]
                else:
                    core, fragment = frags[1], frags[0]
                
                # Get SMILES
                core_smiles = Chem.MolToSmiles(core, canonical=True)
                fragment_smiles = Chem.MolToSmiles(fragment, canonical=True)
                
                # Check fragment size
                fragment_size = (fragment.GetNumHeavyAtoms() - 1) / parent_atom_count 
                if fragment_size > max_fragsize_fraction:
                    continue
                
                # Store data
                fragment_smiles_list.append(fragment_smiles)
                molecule_indices.append(mol_idx)
                core_smiles_list.append(core_smiles)
                cut_indices.append(cut_idx)
                fragment_potencies.append(potency)
                fragment_sizes.append(round(fragment_size, 1))
                parent_smiles_list.append(smiles)
                assay_list.append(assay)
        
        # Create output DataFrame
        if query_or_ref == 'ref':
            fragments_df = pl.DataFrame({
                'id': range(len(fragment_smiles_list)),
                'fragment_smiles': fragment_smiles_list,
                'molecule_idx': molecule_indices, 
                'cut_idx': cut_indices,
                'parent_potency': fragment_potencies,
                'fragment_size': fragment_sizes,
                'parent_smiles': parent_smiles_list,
                'ref_core': core_smiles_list,
                'ref_assay': assay_list
            })
        else:
            fragments_df = pl.DataFrame({
                'id': range(len(fragment_smiles_list)),
                'fragment_smiles': fragment_smiles_list,
                'molecule_idx': molecule_indices, 
                'cut_idx': cut_indices,
                'parent_potency': fragment_potencies,
                'fragment_size': fragment_sizes,
                'parent_smiles': parent_smiles_list,
                'core_smiles': core_smiles_list,
                'assay': assay_list
            })
        
        fragments_df = fragments_df.unique(subset=['parent_smiles', 'fragment_smiles'] + 
                                        (['ref_assay', 'ref_core'] if query_or_ref == 'ref' else ['assay', 'core_smiles']))

        # Store DataFrame
        if query_or_ref == 'query':
            self.query_fragments_df = fragments_df
        else:
            self.fragments_df = fragments_df
            
        return fragments_df

    def query_fragments(self, query_dataset: pd.DataFrame, min_series_length: int = 3, 
                       assay_col: str = 'assay', smiles_col: str = 'smiles', 
                       potency_col: str = 'potency', standardize: bool = True,
                       fragments_already_processed: bool = False,
                       max_fragsize_fraction: float = 0.5) -> pl.DataFrame:
        """
        Query the fragment database using a dataset of molecules.
        
        Parameters
        ----------
        query_dataset : pandas.DataFrame
            Dataset containing query molecules
        min_series_length : int, default=3
            Minimum number of fragments required to consider a series
        assay_col : str, default='assay'
            Name of column containing assay identifiers
        smiles_col : str, default='smiles'
            Name of column containing SMILES strings
        potency_col : str, default='potency'
            Name of column containing potency values
        standardize : bool, default=True
            Whether to standardize molecules using RDKit standardization methods
        max_fragsize_fraction : float, default=0.5
            Maximum allowed size of fragment relative to parent molecule
        fragments_already_processed : bool, default=False
            Whether the input file contains fragments that have already been processed(e.g. originate from this class)
        Returns
        -------
        polars.DataFrame
            DataFrame containing matched series information
        """
        # Get query fragments
        if fragments_already_processed:
            self.query_fragments_df = pl.from_pandas(query_dataset) if not isinstance(query_dataset, pl.DataFrame) else query_dataset
        else:
            self.fragment_molecules(query_dataset, assay_col=assay_col, query_or_ref='query',
                              max_fragsize_fraction=max_fragsize_fraction, standardize=standardize,
                              smiles_col=smiles_col, potency_col=potency_col)

        # Convert to lazy DataFrames
        reference_fragments_lazy = self.fragments_df.lazy()
        query_fragments_lazy = self.query_fragments_df.lazy()

        # Filter and join fragments
        reference_series = (reference_fragments_lazy
            .group_by(['ref_core', 'ref_assay'])
            .agg(pl.n_unique('fragment_smiles').alias('reference_fragment_count'))
            .filter(pl.col('reference_fragment_count') >= min_series_length)
            .join(reference_fragments_lazy, on=['ref_core', 'ref_assay'])
        )

        query_series = (query_fragments_lazy
            .group_by(['core_smiles', 'assay'])
            .agg(pl.n_unique('fragment_smiles').alias('query_fragment_count'))
            .filter(pl.col('query_fragment_count') >= min_series_length)
            .join(query_fragments_lazy, on=['core_smiles', 'assay'])
        )

        # Join and process results
        merged_series = reference_series.join(query_series, on='fragment_smiles')

        matched_series = (merged_series
            .group_by(['ref_core', 'ref_assay', 'core_smiles', 'assay'])  
            .agg([
                pl.n_unique('fragment_smiles').alias('series_length'),
                pl.first('core_smiles').alias('query_core'),
                pl.first('assay').alias('query_assay'),
                pl.col('fragment_smiles').alias('common_fragments').str.join('|'),
                pl.col('parent_potency').cast(pl.Utf8).str.join('|').alias('reference_potencies'),
                pl.col('parent_potency_right').cast(pl.Utf8).str.join('|').alias('query_potencies'),
                (pl.col('parent_potency') * pl.col('parent_potency_right')).sum().alias('potency_dot_product'),
                (pl.col('parent_potency') ** 2).sum().alias('reference_potency_norm_sq'),
                (pl.col('parent_potency_right') ** 2).sum().alias('query_potency_norm_sq'),
                ((pl.col('parent_potency') - pl.col('parent_potency').mean() - 
                (pl.col('parent_potency_right') - pl.col('parent_potency_right').mean()))**2).mean().sqrt().alias('cRMSD')
            ])
            .filter(pl.col('series_length') >= min_series_length)
        )

        # Find additional fragments in reference set not present in query
        unique_reference_fragments = (reference_series
            .join(query_series, on='fragment_smiles', how='left')
            .filter(pl.col('id_right').is_null())
            .select(['fragment_smiles', 'ref_core', 'ref_assay', 'parent_potency'])
            .group_by(['ref_core', 'ref_assay'])
            .agg([
                pl.col('fragment_smiles').str.join('|').alias('new_fragments'),
                pl.col('parent_potency').cast(pl.Utf8).str.join('|').alias('new_fragments_ref_potency')
            ])
            .join(matched_series, on=['ref_core', 'ref_assay'])
            .select([
                'new_fragments', 
                'ref_core', 
                'query_core', 
                'ref_assay', 
                'query_assay', 
                'cRMSD',
                'series_length', 
                'common_fragments',
                'reference_potencies',
                'query_potencies',
                'new_fragments_ref_potency'
            ])
        )
        
        result = unique_reference_fragments.collect().to_pandas()
        result['ref_core_attachment_point'] = result['ref_core'].apply(self._get_attachment_point)
        result['query_core_attachment_point'] = result['query_core'].apply(self._get_attachment_point)
        return result

    
    def _get_attachment_point(self, smiles: str) -> str:
        """
        Get the atom connecting the attachment point [At] to the core structure. Returns None if no
        match is found.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule with attachment point [At]

        Returns
        -------
        str
            SMARTS string of the connecting atom
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            match = mol.GetSubstructMatch(self.attachment_point_substruct)
            return mol.GetAtomWithIdx(match[0]).GetSmarts()
        else:
            return None
        
    def combine_fragments(self, input_df: pd.DataFrame, query_core_col: str = 'query_core', new_fragment_smiles_col: str = 'new_fragments') -> pd.DataFrame: 
        """
        Combine a core structure with a fragment using reaction SMARTS.
        
        Parameters
        ----------
        input dataframe : pandas.DataFrame
            Input dataframe containing 'query_core' and 'new_fragments' columns
        query_core_col : str
            Column name for the query core structure in the input dataframe
        new_fragment_smiles_col : str
            Column name for the new fragments in the input dataframe
            
        Returns
        -------
        pandas.DataFrame
            Input dataframe with 'combined_smiles' column added
        """
        input_df['combined_smiles'] = input_df.apply(lambda row: '|'.join(filter(None, [self._combine_fragment(row[query_core_col], frag) for frag in row[new_fragment_smiles_col].split('|')])), axis=1)
        return input_df



    def _combine_fragment(self, core_smiles: str, fragment_smiles: str) -> Optional[str]:
        """
        Combine a core structure with a fragment using reaction SMARTS.
        
        Parameters
        ----------
        core_smiles : str
            SMILES string of the core structure
        fragment_smiles : str
            SMILES string of the fragment to attach
            
        Returns
        -------
        Optional[str]
            SMILES string of combined molecule if successful,
            None if combination fails
        """
        try:
            core_mol = Chem.MolFromSmiles(core_smiles)
            frag_mol = Chem.MolFromSmiles(fragment_smiles)
            
            if core_mol is None or frag_mol is None:
                return None
                
            products = self.combine_reaction.RunReactants((core_mol, frag_mol))
            
            if not products or not products[0]:
                return None
                
            product = products[0][0]
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product, canonical=True)
            
        except Exception as e:
            logger.error(f"Error combining fragments: {e}")
            return None
