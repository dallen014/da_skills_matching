# %%
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import os


def generate_survey_data(
    num_employees=100, filename="technical_skills_survey.csv", directory="."
):
    """
    Generates a synthetic dataset for a technical skills survey and saves it as a CSV file.

    Parameters:
    num_employees (int): Number of employees for which to generate survey responses (default is 100).
    filename (str): The name of the output CSV file (default is 'technical_skills_survey.csv').
    directory (str): The directory where the file will be saved (default is current directory).

    Returns:
    None
    """
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Define the survey categories
    columns = [
        "AWS_EC2",
        "Azure_VM",
        "Google_Cloud_Functions",
        "Apache_Spark",
        "Apache_Airflow",
        "Databricks",
        "ETL_Tools",
        "Kafka",
        "Spark_Streaming",
        "TensorFlow",
        "PyTorch",
        "Scikit_learn",
        "Regression_Analysis",
        "Time_Series_Analysis",
        "Hypothesis_Testing",
        "Python",
        "Java",
        "SQL",
        "Cloud_Architecture",
        "Microservices",
        "Containerization",
        "Network_Management",
        "Virtualization",
        "Linux_Administration",
        "Cybersecurity",
        "IAM",
        "Firewalls",
        "SQL_Tuning",
        "Database_Replication",
        "Backup_Recovery",
        "User_Research",
        "Prototyping",
        "Design_Tools",
    ]

    # Define the categorical options for the survey
    categories = ["No Knowledge", "Some Knowledge", "Proficient", "Expert"]

    # Generate random responses using the categorical values
    data = np.random.choice(categories, size=(num_employees, len(columns)))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Add an 'Employee_ID' column
    df.insert(0, "Employee_ID", range(1, num_employees + 1))

    # Ensure the directory exists, create it if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Save the DataFrame to the CSV file in the specified directory
    df.to_csv(file_path, index=False)


def cosine_preprocessing(df):
    """
    Preprocess the data for cosine similarity computation.

    The function handles missing values, encodes categorical skill levels into numeric values,
    and returns the preprocessed matrix (excluding Employee_ID).

    Parameters:
    df (pd.DataFrame): The DataFrame containing employee survey responses.

    Returns:
    np.array: A matrix of preprocessed skill ratings (numerical values) ready for cosine similarity computation.
    """

    # Step 1: Check for and handle missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(
            "No Knowledge", inplace=True
        )  # Fill missing values with 'No Knowledge'

    # Step 2: Ordinal Encoding of categorical data (convert "No Knowledge", "Proficient", etc. to numbers)
    categories = ["No Knowledge", "Some Knowledge", "Proficient", "Expert"]

    # Instantiate OrdinalEncoder
    encoder = OrdinalEncoder(
        categories=[categories] * (df.shape[1] - 1)
    )  # Exclude Employee_ID

    # Step 3: Encode the categorical data, excluding Employee_ID
    df_encoded = df.copy()
    df_encoded.iloc[:, 1:] = encoder.fit_transform(df_encoded.iloc[:, 1:])

    # Step 4: Return the encoded matrix (skill ratings) excluding Employee_ID
    return df_encoded.iloc[:, 1:].values


def generate_department_survey(
    directory=".", filename="department_technical_skill.csv"
):
    """
    Generates a synthetic dataset representing ideal skillsets for 10 departments, with tailored proficiency distributions
    for each department based on expected skill needs. Saves the data to a specified directory.

    Parameters:
    directory (str): The directory where the CSV file will be saved (default is current directory).
    filename (str): The name of the CSV file (default is 'department_ideal_skillset.csv').

    Returns:
    pd.DataFrame: A DataFrame containing the skill ratings for each department.
    """

    # Define the department names
    departments = [
        "Data Scientist",
        "Data Engineers",
        "Developers",
        "Analyst",
        "Architects",
        "System Administrators",
        "Security",
        "Database Administrator",
        "UX",
    ]

    # Define the skill categories (expanded to match the employee dataset)
    columns = [
        "AWS_EC2",
        "Azure_VM",
        "Google_Cloud_Functions",
        "Apache_Spark",
        "Apache_Airflow",
        "Databricks",
        "ETL_Tools",
        "Kafka",
        "Spark_Streaming",
        "TensorFlow",
        "PyTorch",
        "Scikit_learn",
        "Regression_Analysis",
        "Time_Series_Analysis",
        "Hypothesis_Testing",
        "Python",
        "Java",
        "SQL",
        "Cloud_Architecture",
        "Microservices",
        "Containerization",
        "Network_Management",
        "Virtualization",
        "Linux_Administration",
        "Cybersecurity",
        "IAM",
        "Firewalls",
        "SQL_Tuning",
        "Database_Replication",
        "Backup_Recovery",
        "User_Research",
        "Prototyping",
        "Design_Tools",
    ]

    # Define the categorical options for the skill levels
    categories = ["No Knowledge", "Some Knowledge", "Proficient", "Expert"]

    # Define the tailored distributions for each department
    skill_profiles = {
        "Data Scientist": {
            "data_science": [
                0.05,
                0.1,
                0.35,
                0.5,
            ],  # Higher likelihood of 'Proficient' or 'Expert'
            "security": [0.6, 0.3, 0.05, 0.05],  # Lower likelihood of security skills
        },
        "Data Engineers": {
            "data_engineering": [0.05, 0.15, 0.4, 0.4],
            "cloud": [0.1, 0.3, 0.35, 0.25],
        },
        "Developers": {
            "programming": [0.05, 0.15, 0.45, 0.35],
            "architecture": [0.2, 0.4, 0.3, 0.1],
        },
        "Analyst": {
            "analysis": [0.05, 0.15, 0.45, 0.35],
            "data_science": [0.2, 0.4, 0.3, 0.1],
        },
        "Architects": {
            "architecture": [0.05, 0.1, 0.35, 0.5],
            "cloud": [0.1, 0.2, 0.4, 0.3],
        },
        "System Administrators": {
            "sysadmin": [0.05, 0.15, 0.35, 0.45],
            "security": [0.2, 0.3, 0.4, 0.1],
        },
        "Security": {
            "security": [0.05, 0.1, 0.35, 0.5],
            "sysadmin": [0.15, 0.3, 0.4, 0.15],
        },
        "Database Administrator": {
            "databases": [0.05, 0.15, 0.4, 0.4],
            "cloud": [0.1, 0.3, 0.4, 0.2],
        },
        "UX": {
            "ux": [0.05, 0.15, 0.45, 0.35],
        },
    }

    # Function to generate a skill profile for a department based on skill group and profile
    def generate_skills(department, category):
        if category in department:
            return np.random.choice(categories, p=department[category])
        else:
            return np.random.choice(
                categories, p=[0.3, 0.4, 0.2, 0.1]
            )  # Default distribution

    # Initialize a list to store department survey responses
    department_data = []

    # Loop through each department and assign tailored skill ratings
    for dept in departments:
        profile = skill_profiles[dept]
        row = [
            generate_skills(profile, "cloud"),  # AWS_EC2
            generate_skills(profile, "cloud"),  # Azure_VM
            generate_skills(profile, "cloud"),  # Google Cloud
            generate_skills(profile, "data_engineering"),  # Apache Spark
            generate_skills(profile, "data_engineering"),  # Apache Airflow
            generate_skills(profile, "data_engineering"),  # Databricks
            generate_skills(profile, "data_engineering"),  # ETL Tools
            generate_skills(profile, "data_engineering"),  # Kafka
            generate_skills(profile, "data_engineering"),  # Spark Streaming
            generate_skills(profile, "data_science"),  # TensorFlow
            generate_skills(profile, "data_science"),  # PyTorch
            generate_skills(profile, "data_science"),  # Scikit-learn
            generate_skills(profile, "analysis"),  # Regression Analysis
            generate_skills(profile, "analysis"),  # Time Series
            generate_skills(profile, "analysis"),  # Hypothesis Testing
            generate_skills(profile, "programming"),  # Python
            generate_skills(profile, "programming"),  # Java
            generate_skills(profile, "programming"),  # SQL
            generate_skills(profile, "architecture"),  # Cloud Architecture
            generate_skills(profile, "architecture"),  # Microservices
            generate_skills(profile, "architecture"),  # Containerization
            generate_skills(profile, "sysadmin"),  # Network Management
            generate_skills(profile, "sysadmin"),  # Virtualization
            generate_skills(profile, "sysadmin"),  # Linux Administration
            generate_skills(profile, "security"),  # Cybersecurity
            generate_skills(profile, "security"),  # IAM
            generate_skills(profile, "security"),  # Firewalls
            generate_skills(profile, "databases"),  # SQL Tuning
            generate_skills(profile, "databases"),  # Database Replication
            generate_skills(profile, "databases"),  # Backup Recovery
            generate_skills(profile, "ux"),  # User Research
            generate_skills(profile, "ux"),  # Prototyping
            generate_skills(profile, "ux"),  # Design Tools
        ]
        department_data.append([dept] + row)

    # Convert to DataFrame
    df_departments = pd.DataFrame(department_data, columns=["Department"] + columns)

    # Ensure the directory exists, create it if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Save the DataFrame to the CSV file in the specified directory
    df_departments.to_csv(file_path, index=False)

    return df_departments
