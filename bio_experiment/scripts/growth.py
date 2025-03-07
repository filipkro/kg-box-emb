import os
import glob
import datetime

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro, levene
# %%
def process_measurement_data(directory_path):
    """
    Process raw measurement data from the BMG Omega Polarstar.

    Input:
        directory_path: Path to the folder containing the files,
        these should be the results files, not the metadata.

    Output:
        data_df: A dataframe of time-series data for each relevant well.
    """
    # Get a list of all files starting with "AUTOMATED" in the directory
    files = glob.glob(os.path.join(directory_path, "AUTOMATED*"))
    
    # Sort the files by modification date (earliest first)
    files.sort(key=lambda x: os.path.getmtime(x))
    
    # List of valid keys from A01 to H12
    valid_keys = [f"{row}{col:02}" for row in 'ABCDEFGH'
                  for col in range(1, 13)]
    
    data_dict = {}
    
    for file_path in files:
        #print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8',
                      errors='ignore') as file:
                inside_data_section = False
                file_time = None  # Initialize file time variable
                
                for line in file:
                    # Extract the date and time from the file
                    if "Date:" in line and "Time:" in line:
                        parts = line.split("Time:")
                        date_part = parts[0].split("Date:")[1].strip()
                        time_part = parts[1].strip()
                        file_time = datetime.datetime.strptime(f"{date_part} "
                                                               f"{time_part}",
                                                    "%d/%m/%Y %H:%M:%S")
                    
                    # Check if we're in the measurement data section
                    if "Measurement Data" in line:
                        inside_data_section = True
                        continue
                    
                    # Skip irrelevant lines
                    if not inside_data_section or line.strip() == "" \
                                    or "=" in line:
                        continue
                    
                    # Extract key-value pairs
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                        except ValueError:
                            continue  # Skip if conversion to float fails
                        
                        # Only add valid keys (A01-H12) to the dictionary
                        if key in valid_keys:
                            if key not in data_dict:
                                data_dict[key] = []
                            data_dict[key].append((file_time, value))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    # Sort the dictionary based on time and convert values back to just floats
    for key in data_dict:
        data_dict[key].sort(key=lambda x: x[0])  # Sort by datetime
        data_dict[key] = [value for _, value in data_dict[key]]
        
    data_df = pd.DataFrame.from_dict(data_dict, orient='index',
                                     columns=[i * 1200
                                              for i in range(len(files))])
    
    return data_df

def unify_well_format(well_name: str) -> str:
    """
    Helper function to unify the formatting of the wells
    Convert strings like 'A01' -> 'A1', 'C08' -> 'C8', etc.
    If the numeric part is all zeros, revert to '0'.
    """
    if not well_name:
        return well_name  # edge case handling
    
    letter = well_name[0]               # e.g. 'A'
    number_str = well_name[1:].lstrip('0')  # e.g. '01' -> '1'
    
    if number_str == "":               # if it was all zeros
        number_str = "0"
    
    return f"{letter}{number_str}"     


def set_datatypes(data, testing_column):
    """
    Properly categorize and order the data so that there are relevant 
    types and baselines for statistical testing. 

    Input
    data: Dataframe with the testing column, treatment and supplementation
                levels from the experimental design, and relevant wells
    testing_column: Name of the tested metric (e.g. AUC, mu, MaxOD)

    Output
    design_df: A dataframe with annotations for each experimental group.
                For use with testing-
    """
    
    rls_data = data[[testing_column, 'Treatment', 'Inositol']].copy()
    
    # Set Treatment as an ordered categorical variable (baseline: 'None')
    rls_data['Treatment'] = pd.Categorical(
        rls_data['Treatment'],
        categories=['None', 'TreatmentLow', 'TreatmentHigh'],
        ordered=True
    )
    
    # Set Inositol as an ordered categorical variable (baseline: 'InositolLow')
    rls_data['Inositol'] = pd.Categorical(
        rls_data['Inositol'],
        categories=['InositolLow', 'InositolMedium', 'InositolHigh'],
        ordered=True
    )
    return rls_data

def create_design_table(df):
    
    """
    Function that generates a design table from the plate/dispensing layout

    Input
    df: A dispensing layout (used with the hamilton liquid handling robot).

    Output
    design_df: A dataframe with annotations for each experimental group.
                For use with testing-
    """
    
    design_df = df.copy()
    
    # Process Supplement Column
    unique_supplement_values = sorted(df['Inositol'].unique())
    supplement_mapping = {0: 'None', unique_supplement_values[1]: 'InositolLow',
                          unique_supplement_values[2]: 'InositolMedium',
                          unique_supplement_values[3]: 'InositolHigh'}
    design_df['Inositol'] = df['Inositol'].map(supplement_mapping)
    
    # Process Treatment Column
    unique_treatment_values = sorted(df['Treatment'].unique())
    treatment_mapping = {0: 'None', unique_treatment_values[1]: 'TreatmentLow',
                         unique_treatment_values[2]: 'TreatmentHigh'}
    design_df['Treatment'] = df['Treatment'].map(treatment_mapping)
    
    # Keep only relevant columns
    design_df = design_df[['Well', 'Description', 'Treatment', 'Inositol']]
    
    return design_df

def compute_auc(df):
    
    """
    Compute AUC for each row's (well's) time series data.

    Input
    df: DataFrame where rows = samples, columns = time points.

    Output
    auc_df: A dataframe with calculated AUC for each well/row
    """
    
    # Exclude the last column (experiment descriptor)
    time_series_data = df
    # Convert column names to time points
    time_points = np.array(time_series_data.columns, dtype=float)  

    auc_results = []
    for well_id, row in time_series_data.iterrows():
        growth_values = row.astype(float).values  # Growth data
        
        # Compute AUC using Trapezoidal Rule
        auc_trapz = np.trapz(growth_values, time_points)
        auc_results.append([well_id, auc_trapz])

    # Create DataFrame with AUC results
    auc_df = pd.DataFrame(auc_results, columns=["Well", "AUC"])
    auc_df.set_index("Well", inplace=True)
    
    return auc_df

def smooth_growth_curves(df, window = 3, center = True):
    """
    Apply a rolling average smoothing to each row's (well's) time series data.

    Input
    df: DataFrame where rows = samples, columns = time points.
    window: Size of the moving window for the rolling average.
    center: If True, the window is centered around each point.
            If False, the window is trailing.

    Output
    df_smoothed: A copy of df with numeric (time) columns smoothed
    """
    
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df_smoothed = df.copy()

    def smooth_row_values(row):
        values = row.values
        n = len(values)

        # Edge handling - extend the series before smoothing
        extended = np.concatenate((values[:window][::-1], values,
                                   values[-window:][::-1]))

        smoothed = pd.Series(extended).rolling(window=window, center=center,
                                               min_periods=1).mean().values
        
        # Ensure smoothed row matches original
        # length so we avoid any unexpected bugs
        start_idx = (len(smoothed) - n) // 2
        smoothed = smoothed[start_idx:start_idx + n]

        return pd.Series(smoothed, index=row.index)

    # Apply smoothing
    df_smoothed[numeric_cols] = \
                df_smoothed[numeric_cols].apply(smooth_row_values, axis=1)

    return df_smoothed


def filter_outlier_growth_curves(df , group_col="Summary",
                                 threshold=1.5) -> pd.DataFrame:
    """
    Remove entire growth curves that are outliers within their experimental group.

    Input
    df: DataFrame where rows = samples,
                        columns = time points + experimental conditions.
    group_col: The column representing experimental groups
                        (e.g., "Summary" or "Description").
    threshold: The threshold for detecting outliers (1.5 for IQR).

    Output
    df_filtered: A filtered DataFrame with outlier growth
                    curves removed based on the given threshold.
    """
    
    # Identify numeric (time-series) columns
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns

    # Compute summary metric per curve (e.g., AUC, total OD)
    df['growth_summary'] = df[numeric_cols].sum(axis=1)  # AUC-like metric

    def detect_outliers(group):
        Q1, Q3 = group['growth_summary'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
        group['outlier'] = (group['growth_summary'] < lower_bound) | \
                                (group['growth_summary'] > upper_bound)
        
        return group

    # Apply outlier detection within each experimental group
    df = df.groupby(group_col, group_keys=False).apply(detect_outliers)

    # Keep only non-outliers
    df_filtered = df[df['outlier'] == False].drop(columns=['growth_summary',
                                                           'outlier'])

    return df_filtered
# %%
BASE = \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)
# %% Load the experimental layout
layout = pd.read_excel(os.path.join(BASE, 'ontEmbeddings_exp/inositol_study/'
                                    'pipetting_layout.xlsx'))

# %% Loading the raw growth data
growth = process_measurement_data(os.path.join(BASE, 'ontEmbeddings_exp/'
                                        'inositol_study/growth_data/NaCl'))
growth.index = growth.index.map(lambda x: unify_well_format(str(x)))
growth_df = growth.merge(layout[['Well','Description']], left_index=True,
                         right_on='Well')

# %% Filter outliers using interquartile range
# (1.5 is a standard threshold)
growth_df_curated = filter_outlier_growth_curves(growth_df, "Description", 1.5)

# %% Subtract the average of the blanks
growth_df_curated.iloc[:,:-2] = growth_df_curated.iloc[:,:-2] - \
    growth_df_curated[growth_df_curated['Description'] ==
                      'Media blank'].mean(numeric_only=True)

# %% Smooth the curves using a rolling median
growth_df_curated = smooth_growth_curves(growth_df_curated, 3, True)

# %% Remove the auxotrophy control as that one is not needed anymore
table_for_testing = growth_df_curated[growth_df_curated['Description'] !=
                                      'Auxotrophy control']

# %% Calculate the AUC of each well
auc_df = compute_auc(table_for_testing.set_index('Well').drop(['Description'],
                                                              axis=1))
auc_df = auc_df.merge(table_for_testing[['Well','Description']],
                      left_index=True, right_on='Well')

# %% Generate the design table from the layout and merge with the data
design_table = create_design_table(layout)
auc_data = design_table.merge(auc_df.drop('Description', axis = 1),
                              left_on='Well', right_on='Well')
auc_data = auc_data[auc_data['Description'] != 'Media blank']

# %% Assume auc_data is your original DataFrame with columns:
#                                               'AUC', 'Treatment', 'Inositol'
# Note that you can use another metric here, e.g. growth rate or max od,
# as long as it is calculated beforehand
testing_column = 'AUC'
categorized_data = set_datatypes(auc_data, testing_column)
categorized_data['group'] = categorized_data['Inositol'].astype(str) + "_" + \
                            categorized_data['Treatment'].astype(str)


#####   Testing part, went with a GLM due to flexibility. #####

# %% Full model (with interaction)
glm_full = smf.glm(formula='AUC ~ C(Inositol) * C(Treatment)',
                   data=categorized_data,
                   family=sm.families.Gaussian()).fit()

# %% Reduced model (without interaction), this to make a
# likelihood test for the interaction later on
glm_no_inter = smf.glm(formula='AUC ~ C(Inositol) + C(Treatment)',
                       data=categorized_data,
                       family=sm.families.Gaussian()).fit()

print("Original GLM (with interaction)")
print(glm_full.summary())



# %% Likelihood Ratio Test for Interaction
LL_full = glm_full.llf
LL_reduced = glm_no_inter.llf
LR_stat = 2 * (LL_full - LL_reduced)
df_diff = glm_full.df_model - glm_no_inter.df_model
p_value_lr = chi2.sf(LR_stat, df_diff)

print("\n Likelihood Ratio Test for Interaction (Original GLM) ")
print(f"LR Statistic: {LR_stat:.4f}")
print(f"Degrees of Freedom Difference: {df_diff}")
print(f"p-value: {p_value_lr:.4f}")



# %% Some diagnostics. Just to make sure our assumptions hold
residuals = glm_full.resid_response
fitted = glm_full.fittedvalues

# %% Normality Diagnostics, if p > 0.05, we're all good
stat, p_shapiro = shapiro(residuals)
print("Shapiro-Wilk Test for Original GLM Residuals:")
print(f"Statistic: {stat:.3f}, p-value: {p_shapiro:.3f}")

# %% Levene's Test for Homoscedasticity, if p > 0.05, we're all good 
unique_groups = categorized_data['group'].unique()
group_residuals = [residuals[categorized_data['group'] == grp]
                   for grp in unique_groups]
levene_stat, p_levene = levene(*group_residuals)
print("\nLevene's Test for Original GLM:")
print(f"Statistic: {levene_stat:.3f}, p-value: {p_levene:.3f}")


# %%
auc = auc_df[auc_df['Description'] != 'Media blank']
auc['treatment'] = auc['Description'].apply(lambda x:
                                x.split(', ')[-1].replace('treatment', 'NaCl'))

auc['Inositol'] = auc['Description'].apply(lambda x: x.split(', ')[0])

# %%
order = ['Low Inositol, No treatment', 'Medium Inositol, No treatment',
         'High Inositol, No treatment', 'Low Inositol, Low treatment',
         'Medium Inositol, Low treatment', 'High Inositol, Low treatment',
         'Low Inositol, High treatment', 'Medium Inositol, High treatment',
         'High Inositol, High treatment']
plt.figure(figsize=(7,5))
sns.boxplot(auc, x='AUC', y='treatment',hue='Inositol',
            hue_order=['Low Inositol', 'Medium Inositol', 'High Inositol'],
            palette='Paired')
plt.xlabel('Area under growth curve', fontsize=12)
plt.ylabel('Treatment levels', fontsize=12)
plt.yticks(rotation=90, fontsize=12, va='center')
plt.xticks([45000, 60000, 75000, 90000, 105000, 120000], fontsize=11)
plt.legend(fontsize=11)
plt.savefig(os.path.join(BASE, 'paper/figs/boxplot_growth.eps'),
            format='eps', bbox_inches='tight')
# %%
full_growth = table_for_testing.drop('Well', axis=1)
fg_m = full_growth.groupby('Description').mean().T
fg_s = full_growth.groupby('Description').std().T
fg_m.index = fg_m.index / 60
fg_s.index = fg_s.index / 60
fg_m.drop('Media blank', axis=1).plot()

# %%
plt.figure()
cm = plt.cm.Paired
for column in fg_m.drop('Media blank', axis=1).columns:
    if 'Low Inositol' in column:
        c = cm(0)
    elif 'Medium Inositol' in column:
        c = cm(1)
    elif 'High Inositol' in column:
        c = cm(2)
    else:
        print(column)
        raise RuntimeError()
    if 'No treatment' in column:
        ls = ':'
    elif 'Low treatment' in column:
        ls = '--'
    elif 'High treatment' in column:
        ls = '-.'
    else:
        print(column)
        raise RuntimeError()
    

    plt.plot(fg_m[column], color=c, linestyle=ls,
             label=column.replace('Inositol', 'Ino').replace('treatment',
                                                             'NaCl'))
plt.legend()
plt.xlabel('Minutes')
plt.ylabel('Optical density')
plt.savefig(os.path.join(BASE, 'paper/figs/growth_curves.eps'),
            format='eps', bbox_inches='tight')

# %%
