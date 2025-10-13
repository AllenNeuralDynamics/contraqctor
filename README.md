[![contraqctor](./assets/logo-letter.svg)](https://allenneuraldynamics.github.io/contraqctor/)

[![contraqctor](https://tinyurl.com/zf46ufwa)](https://allenneuraldynamics.github.io/contraqctor/)
![CI](https://github.com/AllenNeuralDynamics/contraqctor/actions/workflows/contraqctor.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/contraqctor)](https://pypi.org/project/contraqctor/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

# contraqctor

A library for managing data contracts and quality control in behavioral datasets.

> ⚠️ **Caution:**  
> This repository is currently under active development and is subject to frequent changes. Features and APIs may evolve without prior notice.

## Installing and Upgrading

If you choose to clone the repository, you can install the package by running the following command from the root directory of the repository:

```bash
pip install .
```

Otherwise, you can use pip:

```bash
pip install contraqctor
```

## Getting started and API usage

The library provides two main functionalities: data contracts for standardized data loading and quality control tools for data validation.

### Creating and Using Data Contracts

Data contracts provide a standard way to access and load data from various sources. Here's a simple example:

```python
from pathlib import Path
from contraqctor.contract import Dataset, DataStreamCollection
from contraqctor.contract.csv import Csv
from contraqctor.contract.text import Text

# Define the dataset structure
dataset_root = Path("path/to/dataset")
my_dataset = Dataset(
    name="my_dataset",
    version="1.0.0",
    description="Example dataset",
    data_streams=[
        DataStreamCollection(
            name="Behavior",
            description="Behavior data",
            data_streams=[
                Csv(
                    "Position",
                    description="Animal position data",
                    reader_params=Csv.make_params(
                        path=dataset_root / "behavior/position.csv",
                    ),
                ),
                Text(
                    name="Log",
                    description="Session log file",
                    reader_params=Text.make_params(
                        path=dataset_root / "behavior/session.log",
                    ),
                ),
            ],
        ),
    ],
)

# Load a specific stream
position_data = my_dataset["Behavior"]["Position"].load().data
print(f"Position data shape: {position_data.shape}")

# Load all streams and handle errors
my_dataset.load_all()
```

### Quality Control of Primary Data

The QC module helps validate your data to ensure it meets specific requirements:

```python
import contraqctor.qc as qc

# Using the dataset created above
data_stream = my_dataset["Behavior"]["Position"]

# Create and run test suites
runner = qc.Runner()

# Add test suites for different data types
runner.add_suite(qc.csv.CsvTestSuite(data_stream))

# Or create your own custom test suite
class MyCustomTestSuite(qc.Suite):
    def __init__(self, data_stream):
        self.data_stream = data_stream
        
    def test_has_expected_columns(self):
        """Check if data has required columns."""
        expected_cols = {"timestamp", "x", "y", "speed"}
        if not expected_cols.issubset(self.data_stream.data.columns):
            missing = expected_cols - set(self.data_stream.data.columns)
            return self.fail_test(None, f"Missing columns: {missing}")
        return self.pass_test(None, "All required columns present")

runner.add_suite(MyCustomTestSuite(data_stream))

# Run all tests and display results
results = runner.run_all_with_progress()
```

For more detailed examples, please check the [Examples](examples) folder.

---

## Contributors

Contributions to this repository are welcome! However, please ensure that your code adheres to the recommended DevOps practices below:

### Linting

We use [ruff](https://docs.astral.sh/ruff/) as our primary linting tool.

### Testing

Attempt to add tests when new features are added.
To run the currently available tests, run `uv run pytest` from the root of the repository.

### Lock files

We use [uv](https://docs.astral.sh/uv/) to manage our lock files and therefore encourage everyone to use uv as a package manager as well.