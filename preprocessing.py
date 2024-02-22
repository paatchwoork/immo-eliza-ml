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

# encode categorical into numerical
le = LabelEncoder()
for column in cat_features:
    data[column] = le.fit_transform(data[column])

plt.figure()
corrmap = data.corr()[['price']]

#sns.heatmap(corrmap.sort_values(by='price',ascending=False), annot=True)
print(corrmap.sort_values(by='price', ascending=False).head(11).index)
#fig, ax = plt.subplots()
#ax.imshow(corrmap)
#sns.heatmap(data.corr()['price'], cmap="coolwarm")
#plt.show()
