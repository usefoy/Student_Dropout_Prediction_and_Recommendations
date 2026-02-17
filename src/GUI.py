import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import pandas as pd

# Load the model, scaler, and encoder
rf_loaded = joblib.load("dropout_prediction_rf_model.pkl")
scaler_loaded = joblib.load("dropout_prediction_scaler.pkl")
le_loaded = joblib.load("parental_education_encoder.pkl")

# Create the Tkinter window
root = tk.Tk()
root.title("Student Dropout Prediction")
root.geometry("400x400")  # Adjust the window size

# Define the available parental education levels for the dropdown
parental_education_levels = ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]

# Create a function to handle the prediction
def predict_dropout():
    try:
        # Get data from the input fields
        act_score = act_score_slider.get()
        sat_score = sat_score_slider.get()
        parental_education = parental_education_combo.get()
        parental_income = float(parental_income_entry.get())
        high_school_gpa = high_school_gpa_slider.get()
        college_gpa = college_gpa_slider.get()
        years_to_graduate = years_to_graduate_slider.get()
        
        # Prepare the student profile dictionary
        student_profile = {
            'ACT composite score': act_score,
            'SAT total score': sat_score,
            'parental level of education': parental_education,  # Correct column name
            'parental income': parental_income,
            'high school gpa': high_school_gpa,
            'college gpa': college_gpa,
            'years to graduate': years_to_graduate
        }

        # Convert the student profile into a DataFrame
        df_input = pd.DataFrame([student_profile]) 

        # Encode the 'parental level of education' column using the loaded encoder
        df_input['Parental_education_encoded'] = le_loaded.transform(df_input['parental level of education'])

        # Debug: Check the transformed input data
        print(f"Transformed data for prediction:\n{df_input}")

        
        features = ['ACT composite score', 'SAT total score', 'Parental_education_encoded', 'parental income', 
            'high school gpa', 'college gpa', 'years to graduate']
        # Scale the features using the loaded scaler
        df_scaled = scaler_loaded.transform(df_input[features])

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


# ACT Composite Score (Slider)
tk.Label(root, text="ACT Composite Score:").grid(row=0, column=0, padx=10, pady=5)
act_score_slider = tk.Scale(root, from_=0, to=36, orient='horizontal')
act_score_slider.grid(row=0, column=1)

# SAT Total Score (Slider)
tk.Label(root, text="SAT Total Score:").grid(row=1, column=0, padx=10, pady=5)
sat_score_slider = tk.Scale(root, from_=400, to=1600, orient='horizontal')
sat_score_slider.grid(row=1, column=1)

# Parental Level of Education (Dropdown)
tk.Label(root, text="Parental Level of Education:").grid(row=2, column=0, padx=10, pady=5)
parental_education_combo = ttk.Combobox(root, values=parental_education_levels)
parental_education_combo.grid(row=2, column=1)
parental_education_combo.set(parental_education_levels[0])  # Set default value

# Parental Income (Text entry)
tk.Label(root, text="Parental Income:").grid(row=3, column=0, padx=10, pady=5)
parental_income_entry = tk.Entry(root)
parental_income_entry.grid(row=3, column=1)

# High School GPA (Slider)
tk.Label(root, text="High School GPA:").grid(row=4, column=0, padx=10, pady=5)
high_school_gpa_slider = tk.Scale(root, from_=0.0, to=4.0, resolution=0.1, orient='horizontal')
high_school_gpa_slider.grid(row=4, column=1)

# College GPA (Slider)
tk.Label(root, text="College GPA:").grid(row=5, column=0, padx=10, pady=5)
college_gpa_slider = tk.Scale(root, from_=0.0, to=4.0, resolution=0.1, orient='horizontal')
college_gpa_slider.grid(row=5, column=1)

# Years to Graduate (Slider)
tk.Label(root, text="Years to Graduate:").grid(row=6, column=0, padx=10, pady=5)
years_to_graduate_slider = tk.Scale(root, from_=1, to=6, orient='horizontal')
years_to_graduate_slider.grid(row=6, column=1)

# Create the "Predict" button
predict_button = tk.Button(root, text="Predict Dropout Risk", command=predict_dropout)
predict_button.grid(row=7, column=0, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
