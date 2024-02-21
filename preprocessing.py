import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/raw/properties.csv")

print("Proportion of NAN values :")
for column in data.columns:
    nb_nan = data[column].isnull().sum()/data.shape[0]*100
    if nb_nan != 0.0:
        print(f"{column} : {nb_nan}")

cat_features = ['property_type',
        'subproperty_type',
        'region',
        'province',
        'locality',
        'zip_code',
        'equipped_kitchen',
        'state_building',
        'epc',
        'heating_type']

le = LabelEncoder()
for column in cat_features:
    print(column)
    data[column] = le.fit_transform(data[column])
    print(data[column])

plt.figure()
sns.heatmap(data.corr(), cmap="coolwarm")
plt.show()
