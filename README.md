````md
# ML Book Practice — Housing Regression

This repository contains a regression project based on the California Housing dataset.

## Project Structure

```text
datasets/
├── housing/
│   ├── housing.csv
│   └── housing.tgz

final_data/
└── housing_prepared.csv

img/
├── attribute_histogram_plots.png
├── housing_prices_scatterplot.png
└── scatter_matrix_plot.png

models/
├── forest_reg_1.joblib
└── forest_reg_2026-04-17.joblib

Housing.ipynb
main.py
processing_data.py
train_model.py
.gitignore
````

## Data Processing

`processing_data.py`:

* loads dataset from `datasets/housing/housing.csv`
* creates `income_cat` feature for stratified sampling
* splits data using `StratifiedShuffleSplit`
* applies preprocessing pipeline:

  * `SimpleImputer`
  * `StandardScaler`
  * `OneHotEncoder`
  * custom transformer `CombinedAttributesAdder`
* adds features:

  * `rooms_per_household`
  * `population_per_household`
  * `bedrooms_per_room`
* saves processed dataset to `final_data/housing_prepared.csv`

## Model Training

`train_model.py`:

* loads `final_data/housing_prepared.csv`
* splits features and target (`median_house_value`)
* trains `RandomForestRegressor`
* saves trained model to `models/` with current date in filename

## Notebook

`Housing.ipynb` contains experiments and visualizations.

## API (Planned)

`main.py` will be used to implement an API for model inference.

Planned features:

* load trained model
* accept input data via requests
* return predicted house price
* provide endpoints for integration

## How to Run

### Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### Prepare data

```bash
python processing_data.py
```

### Train model

```bash
python train_model.py
```

## Notes

* Dataset: California Housing dataset
* Model: RandomForestRegressor
* All artifacts are stored locally

## Future Improvements

* API implementation in `main.py`
* model evaluation metrics
* configurable pipeline parameters
* inference script for new samples

```