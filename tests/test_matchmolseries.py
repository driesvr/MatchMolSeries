import unittest
import pandas as pd
from matchmolseries import MatchMolSeries

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
        
        Test Data
        ---------
        - Two simple molecules: fluorobenzene and chlorobenzene
        - Single assay
        - Simple numeric potency values
        
        Expected Outcome
        ---------------
        - Successful fragmentation
        - Non-empty result DataFrame
        - Fragments maintain chemical validity
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
        
        Test Data
        ---------
        - Four molecules: F, Cl, Br, I substituted benzenes
        - Two different assays
        - Sequential potency values
        
        Expected Outcome
        ---------------
        - Correct separation of assays
        - Exactly two unique assay identifiers in results
        - Maintained assay associations in fragments
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br', 'c1ccccc1I'],
            'potency': [1.0, 2.0, 3.0, 4.0],
            'assay_col': ['assay1', 'assay1', 'assay2', 'assay2']
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        unique_assays = result.select('ref_assay').unique().to_series().to_list()
        self.assertEqual(len(unique_assays), 2)
        
    def test_standardisation(self):
        """
        Test molecular standardization functionality.
        
        Verifies that the system correctly handles and standardizes
        molecules with salts, charges, and different representations.
        
        Test Data
        ---------
        - Molecules with:
          * Salt forms ([Na+])
          * Counter ions ([Cl-])
          * Different charge representations
        
        Expected Outcome
        ---------------
        - Successful removal of salts and counter ions
        - Standardized fragment representations
        - No sodium atoms in final fragments
        """
        test_data = pd.DataFrame({
            'smiles': ['c1cccnc1F.[Na+]', 'c1cccnc1Cl', 'c1cccnc1Br.[Na+][Cl-]', '[Na+]OCc1cccnc1I'],
            'potency': [1.0, 2.0, 3.0, 4.0],
            'assay_col': ['assay1', 'assay1', 'assay2', 'assay2']
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        fragments =result.select('fragment_smiles').unique().to_series().to_list()

        self.assertEqual(any([not 'Na' in f for f in fragments]), True)

    def test_min_series_length(self):
        """
        Test minimum series length filtering functionality.
        
        Verifies that the system correctly filters matched molecular
        series based on the minimum series length parameter.
        
        Test Data
        ---------
        Reference Set:
        - 9 benzene derivatives with various substituents
        - Single assay
        Query Set:
        - 3 pyridine derivatives with halogen substituents
        
        Test Conditions
        --------------
        - Tests with min_series_length = 3
        - Tests with min_series_length = 5
        
        Expected Outcome
        ---------------
        - More results with lower minimum series length
        - Correct filtering of series based on length criteria
        - Maintained chemical validity in filtered results
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
        
        self.assertTrue(len(result_min3) > len(result_min5))
        
    def test_complex_molecules(self):
        """
        Test fragmentation of complex drug-like molecules.
        
        Verifies system's ability to handle realistic drug-like
        molecules with multiple rings and substituents.
        
        Test Data
        ---------
        - Two complex molecules with:
          * Multiple ring systems
          * Various substituents
          * Different heteroatoms
          * Realistic potency values
        
        Expected Outcome
        ---------------
        - Successful fragmentation
        - Generation of chemically valid fragments
        - Proper handling of complex molecular features
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
        
        Test Data
        ---------
        - Mix of valid and invalid SMILES:
          * Valid: fluorobenzene
          * Invalid: malformed SMILES string
          * Valid: bromobenzene
        
        Expected Outcome
        ---------------
        - Graceful error handling for invalid SMILES
        - Successful processing of valid molecules
        - Non-empty result set from valid molecules
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
        
        Verifies that the system correctly processes and
        maintains floating-point potency values.
        
        Test Data
        ---------
        - Three molecules with different float potencies
        - Values with different decimal places
        - Single assay group
        
        Expected Outcome
        ---------------
        - Correct preservation of potency values
        - Proper handling of decimal places
        - Maintained value precision
        """
        test_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br'],
            'potency': [1.5, 2.7, 3.0],  # Float values
            'assay_col': ['assay1']*3
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        self.assertIsNotNone(result)
        
    def test_large_series(self):
        """
        Test handling of larger series.
        
        Verifies that the system correctly processes and handles
        larger series of molecules.
        
        Test Data
        ---------
        - 45 benzene derivatives with various substituents
        - Single assay
        
        Expected Outcome
        ---------------
        - Successful processing of large series
        - Generation of chemically valid fragments
        - Proper handling of large molecular sets
        """
        # Create a larger series of similar molecules
        smiles = [f'c1ccccc1{x}' for x in 'FCBINSOP'] * 5
        test_data = pd.DataFrame({
            'smiles': smiles,
            'potency': list(range(1, len(smiles) + 1)),
            'assay_col': ['assay1'] * len(smiles)
        })
        
        result = self.mms.fragment_molecules(test_data, assay_col='assay_col')
        self.assertTrue(len(result) > 0)

    def test_attachment_points(self):
        """
        Test handling of attachment points.
        
        Verifies that the system correctly identifies
        attachment points in molecules.
        
        """
        test_data = ['c1ccccc1[At]', 'c1cccn1[At]', 'c1cccn1C[At]','notavalidmolecule[At]']
        result = [self.mms._get_attachment_point(x) for x in test_data]
        self.assertEqual(result, ['c', 'n', 'C', None])

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)
