# Data
Data Cleaning and Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import zscore

# Load the dataset from a CSV file
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found at '{file_path}'.")
        return None

# Step 1: Identify and Handle Missing Data
def handle_missing_data(df):
    print("Step 1: Handling missing data")
    print("Before handling missing data:")
    print(df.head())

    df_cleaned = df.dropna()

    print("\nAfter handling missing data:")
    print(df_cleaned.head())

    return df_cleaned

# Step 2: Remove Duplicates
def remove_duplicates(df):
    print("\nStep 2: Removing duplicates")
    print("Before removing duplicates:")
    print(df.head())

    df_no_duplicates = df.drop_duplicates()

    print("\nAfter removing duplicates:")
    print(df_no_duplicates.head())

    return df_no_duplicates

# Step 3: Detect and Address Outliers
def remove_outliers(df, threshold=3):
    print("\nStep 3: Detecting and addressing outliers")
    # Compute Z-scores for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df[numerical_cols]))

    # Remove rows where any numerical feature has a Z-score greater than the threshold
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]

    print("\nAfter outlier detection and removal:")
    print(df_no_outliers.head())

    return df_no_outliers

# Step 4: Perform Basic Data Transformations
def preprocess_data(input_file, output_file):
    # Load the dataset
    df = load_dataset(input_file)

    if df is not None:
        # Step 1: Handle missing data
        df = handle_missing_data(df)

        # Step 2: Remove duplicates
        df = remove_duplicates(df)

        # Step 3: Detect and address outliers
        df = remove_outliers(df)

        # Step 4a: Scale numerical variables
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if not numerical_cols.empty:
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Save the cleaned and preprocessed dataset
        df.to_csv(output_file, index=False)
        print("\nData cleaning and preprocessing completed. Cleaned dataset saved as '{}'.".format(output_file))

if __name__ == "__main__":
    input_file = "data.csv"  # Change this to your input CSV file
    output_file = "cleaned_data.csv"  # Change this to the desired output CSV file

    preprocess_data(input_file, output_file)

     
Step 1: Handling missing data
Before handling missing data:
  Observation  Y-Kappa  ChipRate  BF-CMratio  BlowFlow  ChipLevel4   \
0    31-00:00    23.10    16.520     121.717  1177.607      169.805   
1    31-01:00    27.60    16.810      79.022  1328.360      341.327   
2    31-02:00    23.19    16.709      79.562  1329.407      239.161   
3    31-03:00    23.60    16.478      81.011  1334.877      213.527   
4    31-04:00    22.90    15.618      93.244  1334.168      243.131   

   T-upperExt-2   T-lowerExt-2    UCZAA  WhiteFlow-4   ...  SteamFlow-4   \
0        358.282         329.545  1.443       599.253  ...        67.122   
1        351.050         329.067  1.549       537.201  ...        60.012   
2        350.022         329.260  1.600       549.611  ...        61.304   
3        350.938         331.142  1.604       623.362  ...        68.496   
4        351.640         332.709    NaN       638.672  ...        70.022   

   Lower-HeatT-3  Upper-HeatT-3   ChipMass-4   WeakLiquorF   BlackFlow-2   \
0        329.432         303.099      175.964      1127.197      1319.039   
1        330.823         304.879      163.202       665.975      1297.317   
2        329.140         303.383      164.013       677.534      1327.072   
3        328.875         302.254      181.487       767.853      1324.461   
4        328.352         300.954      183.929       888.448      1343.424   

   WeakWashF   SteamHeatF-3   T-Top-Chips-4   SulphidityL-4   
0     257.325         54.612         252.077             NaN  
1     241.182         46.603         251.406           29.11  
2     237.272         51.795         251.335             NaN  
3     239.478         54.846         250.312           29.02  
4     215.372         54.186         249.916           29.01  

[5 rows x 23 columns]

After handling missing data:
  Observation  Y-Kappa  ChipRate  BF-CMratio  BlowFlow  ChipLevel4   \
1    31-01:00    27.60    16.810      79.022  1328.360      341.327   
3    31-03:00    23.60    16.478      81.011  1334.877      213.527   
5     1-08:00    14.23    15.350      85.518  1171.604      198.538   
7    31-06:00    22.65    14.100      91.887  1307.852      288.989   
9    31-08:00    24.70    13.850      96.208  1334.892      362.511   

   T-upperExt-2   T-lowerExt-2    UCZAA  WhiteFlow-4   ...  SteamFlow-4   \
1        351.050         329.067  1.549       537.201  ...        60.012   
3        350.938         331.142  1.604       623.362  ...        68.496   
5        344.014         325.195  1.436       628.245  ...        65.225   
7        352.321         331.162  1.468       625.549  ...        71.298   
9        352.372         327.358  1.515       553.172  ...        64.249   

   Lower-HeatT-3  Upper-HeatT-3   ChipMass-4   WeakLiquorF   BlackFlow-2   \
1        330.823         304.879      163.202       665.975      1297.317   
3        328.875         302.254      181.487       767.853      1324.461   
5        322.103         298.517      165.814       826.243       907.641   
7        329.662         301.539      179.886       837.178      1315.111   
9        332.264         305.419      166.120       909.810      1318.725   

   WeakWashF   SteamHeatF-3   T-Top-Chips-4   SulphidityL-4   
1     241.182         46.603         251.406           29.11  
3     239.478         54.846         250.312           29.02  
5     595.875         52.807         249.580           30.34  
7     234.047         53.805         249.971           29.22  
9     180.375         48.842         251.121           29.21  

[5 rows x 23 columns]

Step 2: Removing duplicates
Before removing duplicates:
  Observation  Y-Kappa  ChipRate  BF-CMratio  BlowFlow  ChipLevel4   \
1    31-01:00    27.60    16.810      79.022  1328.360      341.327   
3    31-03:00    23.60    16.478      81.011  1334.877      213.527   
5     1-08:00    14.23    15.350      85.518  1171.604      198.538   
7    31-06:00    22.65    14.100      91.887  1307.852      288.989   
9    31-08:00    24.70    13.850      96.208  1334.892      362.511   

   T-upperExt-2   T-lowerExt-2    UCZAA  WhiteFlow-4   ...  SteamFlow-4   \
1        351.050         329.067  1.549       537.201  ...        60.012   
3        350.938         331.142  1.604       623.362  ...        68.496   
5        344.014         325.195  1.436       628.245  ...        65.225   
7        352.321         331.162  1.468       625.549  ...        71.298   
9        352.372         327.358  1.515       553.172  ...        64.249   

   Lower-HeatT-3  Upper-HeatT-3   ChipMass-4   WeakLiquorF   BlackFlow-2   \
1        330.823         304.879      163.202       665.975      1297.317   
3        328.875         302.254      181.487       767.853      1324.461   
5        322.103         298.517      165.814       826.243       907.641   
7        329.662         301.539      179.886       837.178      1315.111   
9        332.264         305.419      166.120       909.810      1318.725   

   WeakWashF   SteamHeatF-3   T-Top-Chips-4   SulphidityL-4   
1     241.182         46.603         251.406           29.11  
3     239.478         54.846         250.312           29.02  
5     595.875         52.807         249.580           30.34  
7     234.047         53.805         249.971           29.22  
9     180.375         48.842         251.121           29.21  

[5 rows x 23 columns]

After removing duplicates:
  Observation  Y-Kappa  ChipRate  BF-CMratio  BlowFlow  ChipLevel4   \
1    31-01:00    27.60    16.810      79.022  1328.360      341.327   
3    31-03:00    23.60    16.478      81.011  1334.877      213.527   
5     1-08:00    14.23    15.350      85.518  1171.604      198.538   
7    31-06:00    22.65    14.100      91.887  1307.852      288.989   
9    31-08:00    24.70    13.850      96.208  1334.892      362.511   

   T-upperExt-2   T-lowerExt-2    UCZAA  WhiteFlow-4   ...  SteamFlow-4   \
1        351.050         329.067  1.549       537.201  ...        60.012   
3        350.938         331.142  1.604       623.362  ...        68.496   
5        344.014         325.195  1.436       628.245  ...        65.225   
7        352.321         331.162  1.468       625.549  ...        71.298   
9        352.372         327.358  1.515       553.172  ...        64.249   

   Lower-HeatT-3  Upper-HeatT-3   ChipMass-4   WeakLiquorF   BlackFlow-2   \
1        330.823         304.879      163.202       665.975      1297.317   
3        328.875         302.254      181.487       767.853      1324.461   
5        322.103         298.517      165.814       826.243       907.641   
7        329.662         301.539      179.886       837.178      1315.111   
9        332.264         305.419      166.120       909.810      1318.725   

   WeakWashF   SteamHeatF-3   T-Top-Chips-4   SulphidityL-4   
1     241.182         46.603         251.406           29.11  
3     239.478         54.846         250.312           29.02  
5     595.875         52.807         249.580           30.34  
7     234.047         53.805         249.971           29.22  
9     180.375         48.842         251.121           29.21  

[5 rows x 23 columns]

Step 3: Detecting and addressing outliers

After outlier detection and removal:
  Observation  Y-Kappa  ChipRate  BF-CMratio  BlowFlow  ChipLevel4   \
1    31-01:00    27.60    16.810      79.022  1328.360      341.327   
3    31-03:00    23.60    16.478      81.011  1334.877      213.527   
5     1-08:00    14.23    15.350      85.518  1171.604      198.538   
7    31-06:00    22.65    14.100      91.887  1307.852      288.989   
9    31-08:00    24.70    13.850      96.208  1334.892      362.511   

   T-upperExt-2   T-lowerExt-2    UCZAA  WhiteFlow-4   ...  SteamFlow-4   \
1        351.050         329.067  1.549       537.201  ...        60.012   
3        350.938         331.142  1.604       623.362  ...        68.496   
5        344.014         325.195  1.436       628.245  ...        65.225   
7        352.321         331.162  1.468       625.549  ...        71.298   
9        352.372         327.358  1.515       553.172  ...        64.249   

   Lower-HeatT-3  Upper-HeatT-3   ChipMass-4   WeakLiquorF   BlackFlow-2   \
1        330.823         304.879      163.202       665.975      1297.317   
3        328.875         302.254      181.487       767.853      1324.461   
5        322.103         298.517      165.814       826.243       907.641   
7        329.662         301.539      179.886       837.178      1315.111   
9        332.264         305.419      166.120       909.810      1318.725   

   WeakWashF   SteamHeatF-3   T-Top-Chips-4   SulphidityL-4   
1     241.182         46.603         251.406           29.11  
3     239.478         54.846         250.312           29.02  
5     595.875         52.807         249.580           30.34  
7     234.047         53.805         249.971           29.22  
9     180.375         48.842         251.121           29.21  

[5 rows x 23 columns]

Data cleaning and preprocessing completed. Cleaned dataset saved as 'cleaned_data.csv'.
