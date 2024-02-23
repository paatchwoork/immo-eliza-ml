# Model card

## Project context

The company Immo-Eliza contacted us to build a predictive model about the housing prices in Belgium taking into account many features. The first step of the project has been to scrape ImmoWeb for the features and to clean the dataset.
The present model builds on that datasets and aims to predict the price of a property based on a chosen set of characteristics.

## Data

The predictive model takes as input the following features:

Property type
Subproperty type
Region
Province
Locality
Zip code
Latitude
Longitude
Construction year
Total area sqm
Surface land sqm
Nbr frontages
Nbr bedrooms
Equipped kitchen
Furnished
Open fire
Terrace
Terrace sqm
Garden
Garden sqm
Swimming pool
Floodzone
State building,
Primary energy consumption sqm
Epc
Heating type
Double glazing
Cadastral income

And returns the price.

## Model details

Two models have been trained : Linear Regression and Linear Model trained with L1 prior as regularizer (aka the Lasso).

to choose the model, please use the ```-m``` flag followed by the name of the model (LR for linear regression or LA for Lasso)

The Lasso predictive model uses as parameters alpha of 1000 and a random coefficient selection.

The missing values are imputed using a 'most frequent' startegy.
The categorical features are encoded using the One Hot encoding method.
Outliers have been generously removed using the InterQuartile Range method in orger to keep just the strongest correlations.

## Performance

The main performance metric used is the R-Squared. The trained models also come with a residual plot graph to asses the bias (or lack thereof) of the model.
On the test sample, the models give the following results :

Linear Regression
Train R² score: 0.5035045877855528
Test R² score: 0.547594984729892

Lasso
Train R² score: 0.47064380486826485
Test R² score: 0.5262099281126753

## Limitations

The R-Squared metrics are generally not as high as the team originally wanted them, which means that the predictions can generally fall a bit short. In the future more models will be tested and implemented.

## Usage

All dependencies are listed in the requirements.txt file.
It is advisable to use a virtual environment :
```python -m venv``` to create
```pip install -r requirements.txt``` to install
```source ./.venv/bin/activate``` to use

To run the predictions:
```python predict.py -i input_file -o output_file -m model```

## Maintainers

Me ! I'm Léa Konincks *.*

konincks.lea@gmail.com
