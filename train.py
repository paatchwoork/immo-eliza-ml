import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from yellowbrick.regressor import ResidualsPlot

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/pproc/properties.csv")

    # Define features to use
    num_features = ['nbr_bedrooms',#, 
            'total_area_sqm',#, 
            'latitude', 
            'surface_land_sqm', 
            'nbr_frontages', 
            'terrace_sqm', 
            'garden_sqm'
            ]
    fl_features = ['fl_swimming_pool',]
    cat_features = ['region','property_type' , 'subproperty_type']

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Standardize the numerical values
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    # Find the outliers with elliptic envelope
    # 1 are inliers, -1 are outliers
    #ee = EllipticEnvelope(random_state=0).fit())
    #X_train_out = ee.predict(X_train)
    #X_train = X_train.drop(X_train[X_train_out == -1].index).reset_index(drop=True)
    #y_train = y_train.drop(y_train[X_train_out == -1].index).reset_index(drop=True)

    # Train the model
    model = LinearRegression()
    #model = LogisticRegression()
    model.fit(X_train, y_train)

    #poly = PolynomialFeatures(degree=3, include_bias=False)
    #poly_features = poly.fit_transform(X_train)
    #model = LinearRegression()
    #model.fit(poly_features, y_train)
    #model = PolynomialFeatures()


    # Evaluate the model
    # R2 evaluation
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Residual plot evaluation
    visualizer = ResidualsPlot(model,train_alpha=0.5, test_alpha=0.5) # Initialize the residual plot visualizer
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    model_name = f'{model=}'.split('=')[1]
    visualizer.show(outpath=f"./eval/residual_plot/{model_name[:-2]}.png")                 # Finalize and render the figure

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, f"models/{model_name[:-2]}.joblib")


if __name__ == "__main__":
    train()
