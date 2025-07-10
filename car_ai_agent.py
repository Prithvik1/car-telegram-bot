import pandas as pd
import numpy as np
import pickle
import os
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


os.environ["GOOGLE_API_KEY"] = "AIzaSyCmfOTdnJQlRSHK_2tZaDvlfmIekc08T8c"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def load_data():
    df = pd.read_csv("car_data.csv")
    df.columns = df.columns.str.strip()
    df["Ex-Showroom-Price"] = df["Ex-Showroom-Price"].replace('[₹,]', '', regex=True).astype(int)
    return df


def preprocess_data(df):
    y = df["Ex-Showroom-Price"]
    X = df.drop("Ex-Showroom-Price", axis=1)
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[cat_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
    X_final = pd.concat([X_cat_df, X[num_cols].reset_index(drop=True)], axis=1)
    return X_final, y, encoder, cat_cols, num_cols


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae


def map_needs_to_type(need):
    need = need.lower()
    if "family" in need or "kids" in need:
        return "SUV|MPV"
    elif "sport" in need or "fast" in need:
        return "Coupe|Sedan"
    elif "budget" in need or "cheap" in need:
        return "Hatchback"
    elif "offroad" in need or "rugged" in need:
        return "SUV"
    elif "luxury" in need or "premium" in need:
        return "Sedan|Coupe|SUV"
    else:
        return "Sedan|SUV|Hatchback|MPV"


def suggest_cars(df):
    print("\n🧠 Let’s understand your needs...\n")

    car_need = input("🚘 What type of car are you looking for?").strip()
    fuel_type = input("⛽ Preferred fuel type? (Petrol, Diesel, Hybrid, Electric): ").strip().capitalize()
    year = int(input("📅 Minimum year of manufacture you want (e.g., 2020): ").strip())

    car_type_regex = map_needs_to_type(car_need)

    matches = df[
        (df["Type"].str.contains(car_type_regex, case=False, na=False)) &
        (df["Fuel Type"].str.lower() == fuel_type.lower()) &
        (df["Year"] >= year)
    ]

    if matches.empty:
        print("\n❌ No cars found matching your preferences.")
        return None

    print(f"\n✅ Found {len(matches)} matching cars:\n")
    for i, row in matches.iterrows():
        print(f"{i}. {row['Brand']} {row['Model']} ({row['Year']}, {row['Type']}, {row['Fuel Type']})")

    selected_index = int(input("\n🎯 Enter the number of the car you want to select: ").strip())
    if selected_index not in matches.index:
        print("❌ Invalid selection.")
        return None

    return matches.loc[selected_index]


def ask_gemini_to_estimate(car_details, ml_price):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are a car pricing expert.

Here is the car selected by the user:
- Brand: {car_details['Brand']}
- Model: {car_details['Model']}
- Year: {car_details['Year']}
- Type: {car_details['Type']}
- Fuel Type: {car_details['Fuel Type']}
- Original Ex-Showroom Price: ₹{car_details['Ex-Showroom-Price']:,}

A machine learning model has estimated the current resale value at ₹{int(ml_price):,}.

Based on this, please respond with only your refined resale price in INR — **just the number**, no currency symbol, no explanation.
"""

    response = model.generate_content(prompt)
    print("\n🤖 Gemini says:\n")
    print(response.text)


def predict_selected_car(model, encoder, car, cat_cols, num_cols):
    input_data = car.drop("Ex-Showroom-Price")
    input_df = pd.DataFrame([input_data])

    user_encoded = encoder.transform(input_df[cat_cols])
    user_encoded_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(cat_cols))
    user_final_df = pd.concat([user_encoded_df, input_df[num_cols].reset_index(drop=True)], axis=1)

    prediction = model.predict(user_final_df)[0]
    ask_gemini_to_estimate(car, prediction)

if __name__ == "__main__":
    print("📥 Loading data and training model...")
    df = load_data()
    X, y, encoder, cat_cols, num_cols = preprocess_data(df)
    model, mae = train_model(X, y)


    selected_car = suggest_cars(df)
    if selected_car is not None:
        predict_selected_car(model, encoder, selected_car, cat_cols, num_cols)