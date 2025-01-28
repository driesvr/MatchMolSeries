# MatchMolSeries (Matched Molecular Series)

A Python package for analyzing matched molecular series and proposing new molecules based on existing series with similar SAR.

## Features

- Molecular fragmentation using SMARTS-based transformations
- Fragment database creation and querying
- Series matching with customizable parameters
- Efficient data handling with Polars DataFrames
- RDKit-based chemical structure processing

## Installation

```bash
git clone https://github.com/driesvr/matchmolseries.git
cd matchmolseries
pip install .
```

For development installation:

```bash
git clone https://github.com/driesvr/matchmolseries.git
cd matchmolseries
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- RDKit >= 2022.9.1
- Pandas >= 1.3.0
- Polars >= 0.18.0
- NumPy >= 1.20.0

## Quick Start

```python
from matchmolseries import MatchMolSeries
import pandas as pd

# Initialize MatchMolSeries
mms = MatchMolSeries()

# Prepare your data
data = pd.DataFrame({
    'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br'],
    'potency': [1.0, 2.0, 3.0],
    'assay': ['assay1', 'assay1', 'assay1']
})

# Fragment reference molecules
fragments = mms.fragment_molecules(data)

# Optional: save fragments to parquet and load them in again
mms.save_fragments('fragments.parquet')
mms.load_fragments('fragments.parquet')


# Query with new molecules
query_data = pd.DataFrame({
    'smiles': ['c1cnccc1F', 'c1cnccc1Br'],
    'potency': [1.5, 2.5],
    'assay': ['assay1', 'assay1']
})

# Find matching series
matches = mms.query_fragments(query_data, min_series_length=2)

# stitch fragments together into full molecules
matches = mms.combine_fragments(matches)

```

## Documentation

### MatchMolSeries Class

The main class for molecular fragmentation and analysis.

#### Methods

- `fragment_molecules(input_df, ...)`: Fragment molecules from input DataFrame
- `query_fragments(query_dataset, ...)`: Query fragment database
- `combine_fragments(core_smiles, fragment_smiles)`: Combine core and fragment
- `load_fragments(path)`: Load fragment database
- `save_fragments(path)`: Save fragment database
- `combine_fragments`: combine core with fragments into full molecules


### Fragment Database

The fragment dataset is stored in memory for rapid access. It can be saved as a Parquet or csv file. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
The MMS method was originally introduced by Wawer and Bajorath. Ehmki and Kramer recommended the cRMSD metric for assessing series similarity. O'Boyle et al. proposed the fragmentation patterns used herein.
- Wawer and Bajorath: [Local Structural Changes, Global Data Views: Graphical Substructure−Activity Relationship Trailing](https://pubs.acs.org/doi/10.1021/jm200026b)
- Ehmki and Kramer: [Matched Molecular Series: Measuring SAR Similarity](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00709)
- O'Boyle et al: [Using Matched Molecular Series as a Predictive Tool To Optimize Biological Activity](https://pubs.acs.org/doi/10.1021/jm500022q)
