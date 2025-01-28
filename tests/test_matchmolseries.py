import unittest
import pandas as pd
from matchmolseries import MatchMolSeries
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

class TestMatchMolSeries(unittest.TestCase):
    def setUp(self):
        """
        Test setup method.
        Creates a fresh MatchMolSeries instance for each test to ensure
        a clean state.
        """
        self.mms = MatchMolSeries()
        
    def test_basic_fragmentation(self):
        """
        Test basic molecule fragmentation functionality.
        
        Tests fragmentation of simple aromatic molecules with different
        halogen substituents to verify basic functionality.
        
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl'],
            'potency': [1.0, 2.0],
            'assay_col': ['assay1', 'assay1']
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        
    def test_multiple_assays(self):
        """
        Test handling of molecules from multiple assays.
        
        Verifies that the system correctly processes and organizes
        fragments from molecules belonging to different assays.
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br', 'c1ccccc1I'],
            'potency': [1.0, 2.0, 3.0, 4.0],
            'assay_col': ['assay1', 'assay1', 'assay2', 'assay2']
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        unique_assays = result.select('ref_assay').unique().to_series().to_list()
        self.assertEqual(len(unique_assays), 2)
        
    def test_min_series_length(self):
        """
        Test minimum series length filtering functionality.
        
        Verifies that the system correctly filters matched molecular
        series based on the minimum series length parameter.
        
        """
        ref_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br','c1ccccc1N', 'c1ccccc1O', 'c1ccccc1CF','c1ccccc1NC', 'c1ccccc1CO', 'c1ccccc1OC(F)(F)F'],
            'potency': [1.0, 2.0, 3.0]*3,
            'assay_col': ['assay1']*9
        })
        
        query_data = pd.DataFrame({
            'smiles': ['c1cnccc1F', 'c1cnccc1Cl', 'c1cnccc1Br'],
            'potency': [1.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })
        
        self.mms.fragment_molecules(ref_data, assay_col='assay_col', query_or_ref='ref')
        result_min3 = self.mms.query_fragments(query_data, min_series_length=3, assay_col='assay_col')
        result_min5 = self.mms.query_fragments(query_data, min_series_length=5, assay_col='assay_col')
        
        self.assertListEqual([len(result_min3), len(result_min5)], [1 , 0])
        
    def test_complex_molecules(self):
        """
        Test fragmentation of complex drug-like molecules.
        
        Verifies system's ability to handle realistic drug-like
        molecules with multiple rings and substituents.
        """
        test_data = pd.DataFrame({
            'smiles': [
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5',
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=N5'
            ],
            'potency': [7.5, 8.2],
            'assay_col': ['assay1', 'assay1']
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        
    def test_invalid_smiles(self):
        """
        Test system's handling of invalid SMILES strings.
        
        Verifies that the system gracefully handles invalid
        input while continuing to process valid molecules.
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'invalid_smiles', 'c1ccccc1Br'],
            'potency': [1.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        # Should still process valid molecules
        self.assertTrue(len(result) > 0)
        
    def test_potency_values(self):
        """
        Test handling of different potency value types.
        
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br'],
            'potency': [1.5, 2.7, 3],  # Float values
            'assay_col': ['assay1']*3
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        self.assertTrue(len(result) > 0)

    def test_attachment_points(self):
        """
        Test handling of attachment points.
                
        """
        test_data = ['c1ccccc1[At]', 'c1cccn1[At]', 'c1cccn1C[At]','notavalidmolecule[At]']
        result = [self.mms._get_attachment_point(x) for x in test_data]
        self.assertEqual(result, ['c', 'n', 'C', None])

    def test_standardisation(self):
        """
        Test standardisation of molecules.

        Verifies that the system correctly standardises
        molecules based on the specified standardisation method.
        
        """
        frag_remover = rdMolStandardize.FragmentRemover()

        test_data1 = [Chem.MolFromSmiles(x) for x in ['c1ccccc1', 'c1cccnc1.Cl', 'c1cccnc1.CC(=O)O', 'CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2']]
        test_data2 = [Chem.MolFromSmiles(x) for x in ['C1=CC=CC=C1', 'c1cccnc1', 'c1cccnc1', 'C([C@H]1[C@@](OC(CC)=O)(c2ccccc2)CC[NH+](C)C1)C=C']]
        test_data1_standardised = [Chem.MolToSmiles(self.mms._standardize_mol(x, frag_remover)) for x in test_data1]
        test_data2_standardised = [Chem.MolToSmiles(self.mms._standardize_mol(x, frag_remover)) for x in test_data2]
        self.assertListEqual(test_data1_standardised,test_data2_standardised)

    def test_concatenation_order(self):
        """
        Test concatenation order.

        Verifies that the system correctly concatenates
        molecules and their respective potency values
        
        """
        ref_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br','c1ccccc1N', 'c1ccccc1OC(F)(F)F'],
            'potency': [1.0, 2.0, 3.0, 4.0, 5.0],
            'assay_col': ['assay1']*5
        })
        
        query_data = pd.DataFrame({
            'smiles': ['c1cnccc1F', 'c1cnccc1Cl', 'c1cnccc1Br'],
            'potency': [1.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })
        self.mms.fragment_molecules(ref_data, assay_col='assay_col', query_or_ref='ref')
        result = self.mms.query_fragments(query_data, min_series_length=3, assay_col='assay_col')
        new_frags = result.new_fragments[0].split('|')
        ref_potency = result.new_fragments_ref_potency[0].split('|')
        self.assertEqual(new_frags.index('N[At]'), ref_potency.index('4.0')) 
        self.assertEqual(new_frags.index('FC(F)(F)O[At]'), ref_potency.index('5.0')) 

    def test_cRMSD(self):
        """
        Test cRMSD calculation.
        
        """
        ref_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br','c1ccccc1N', 'c1ccccc1OC(F)(F)F'],
            'potency': [1.0, 2.0, 3.0, 4.0, 5.0],
            'assay_col': ['assay1']*5
        })
        
        query_data = pd.DataFrame({
            'smiles': ['c1cnccc1F', 'c1cnccc1Cl', 'c1cnccc1Br'],
            'potency': [1.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })

        query_data2 = pd.DataFrame({
            'smiles': ['c1cnccc1F', 'c1cnccc1Cl', 'c1cnccc1Br'],
            'potency': [2.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })

        self.mms.fragment_molecules(ref_data, assay_col='assay_col', query_or_ref='ref')
        result = self.mms.query_fragments(query_data, min_series_length=3, assay_col='assay_col')
        result2 = self.mms.query_fragments(query_data2, min_series_length=3, assay_col='assay_col')
        self.assertAlmostEqual(result.cRMSD[0], 0.0)
        self.assertAlmostEqual(result2.cRMSD[0], 0.47140452078125)

        
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)
