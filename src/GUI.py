import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Importing ttk for Combobox
import joblib
import pandas as pd

# Load the model, scaler, and encoder
try:
    rf_loaded = joblib.load("dropout_prediction_rf_model.pkl")
    scaler_loaded = joblib.load("dropout_prediction_scaler.pkl")
    le_loaded = joblib.load("parental_education_encoder.pkl")
    print("Model, scaler, and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoder: {e}")

# Define the feature names (same as used in the model)
features = ['ACT composite score', 'SAT total score', 'Parental level of education', 'Parental income',
            'High school GPA', 'College GPA', 'Years to graduate', 'Parental_education_encoded']

# Create the Tkinter window
root = tk.Tk()
root.title("Student Dropout Prediction")

# Create a function to handle the prediction
def predict_dropout():
    try:
        # Get data from the input fields
        act_score = float(act_score_entry.get())
        sat_score = float(sat_score_entry.get())
        parental_education = parental_education_combobox.get()  # Getting the selected value
        parental_income = float(parental_income_entry.get())
        high_school_gpa = float(high_school_gpa_entry.get())
        college_gpa = float(college_gpa_entry.get())
        years_to_graduate = float(years_to_graduate_entry.get())
        
        # Debug: Print inputs to ensure they are captured correctly
        print(f"Inputs: ACT Score: {act_score}, SAT Score: {sat_score}, Parental Education: {parental_education}, "
              f"Parental Income: {parental_income}, High School GPA: {high_school_gpa}, College GPA: {college_gpa}, "
              f"Years to Graduate: {years_to_graduate}")

        # Prepare the student profile dictionary
        student_profile = {
            'ACT composite score': act_score,
            'SAT total score': sat_score,
            'Parental level of education': parental_education,
            'Parental income': parental_income,
            'High school GPA': high_school_gpa,
            'College GPA': college_gpa,
            'Years to graduate': years_to_graduate
        }

        # Convert the student profile into a DataFrame
        df_input = pd.DataFrame([student_profile])

        # Encode the 'Parental level of education' column using the loaded encoder
        df_input['Parental_education_encoded'] = le_loaded.transform(df_input['Parental level of education'])

        # Debug: Check the transformed input data
        print(f"Transformed data for prediction:\n{df_input}")

        # Scale the features using the loaded scaler
        df_scaled = scaler_loaded.transform(df_input[features[:-1]])  # Exclude 'Parental_education_encoded' from scaling

        # Debug: Check scaled data
        print(f"Scaled data:\n{df_scaled}")

        # Make a prediction using the loaded model
        dropout_prob = rf_loaded.predict_proba(df_scaled)[0][1]
        prediction = rf_loaded.predict(df_scaled)[0]

        # Show the result in the GUI
        messagebox.showinfo("Prediction Result", 
                            f"Dropout Probability: {dropout_prob:.2f}\nPrediction: {'Dropout' if prediction == 1 else 'No Dropout'}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error: {e}")

# Create the form labels and input fields
tk.Label(root, text="ACT Composite Score:").grid(row=0, column=0)
act_score_entry = tk.Entry(root)
act_score_entry.grid(row=0, column=1)

tk.Label(root, text="SAT Total Score:").grid(row=1, column=0)
sat_score_entry = tk.Entry(root)
sat_score_entry.grid(row=1, column=1)

tk.Label(root, text="Parental Level of Education:").grid(row=2, column=0)
parental_education_combobox = ttk.Combobox(root, values=["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"])
parental_education_combobox.grid(row=2, column=1)

tk.Label(root, text="Parental Income:").grid(row=3, column=0)
parental_income_entry = tk.Entry(root)
parental_income_entry.grid(row=3, column=1)

tk.Label(root, text="High School GPA:").grid(row=4, column=0)
high_school_gpa_entry = tk.Entry(root)
high_school_gpa_entry.grid(row=4, column=1)

tk.Label(root, text="College GPA:").grid(row=5, column=0)
college_gpa_entry = tk.Entry(root)
college_gpa_entry.grid(row=5, column=1)

tk.Label(root, text="Years to Graduate:").grid(row=6, column=0)
years_to_graduate_entry = tk.Entry(root)
years_to_graduate_entry.grid(row=6, column=1)

# Create the "Predict" button
predict_button = tk.Button(root, text="Predict Dropout Risk", command=predict_dropout)
predict_button.grid(row=7, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()
