import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# File path for the dataset
data_file_path = "/Users/Mathieu/Documents/Research Mathieu/Zinsstruktur etc Papi/Zinsstruktur/Base_Streamlit01.csv"

# Load the dataset
df = pd.read_csv(data_file_path)

# Page configuration
st.set_page_config(page_title="PCA Analysis", layout="wide")


# Function for performing PCA and visualizations
def perform_pca_analysis_streamlit(df, selected_features):
    """
    Streamlit version of perform_pca_analysis to display outputs in the app.
    """
    # Step 1: Handle the 'day' column
    if 'day' in df.columns:
        dates = pd.to_datetime(df['day'])
        df_features = df[selected_features]
    else:
        st.error("The 'day' column is missing. Ensure it exists for plotting.")
        return

    # Step 2: Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # Step 3: Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)

    # Step 4: Explained Variance and Scree Plot
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    st.subheader("Explained Variance Ratio")
    st.write(pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance": explained_variance,
        "Cumulative Variance": cumulative_variance
    }))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xlabel('Principal Component Index')
    ax.set_title('Scree Plot')
    ax.legend(loc='best')
    st.pyplot(fig)

    # Step 5: PCA Loadings
    loadings = pd.DataFrame(
        pca.components_,
        columns=df_features.columns,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )
    st.subheader("PCA Loadings")
    st.write(loadings)

    # Step 6: Biplot
    pc_scores = pca_result[:, :2]
    pc_loadings = pca.components_[:2, :]
    scaling_factor = np.max(np.abs(pc_scores)) / np.max(np.abs(pc_loadings))
    pc_loadings_scaled = pc_loadings * scaling_factor

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.5, label='Data Projections')
    for i, var in enumerate(df_features.columns):
        ax.arrow(0, 0, pc_loadings_scaled[0, i], pc_loadings_scaled[1, i],
                 color='red', alpha=0.75, head_width=0.02 * scaling_factor, head_length=0.05 * scaling_factor)
        ax.text(pc_loadings_scaled[0, i] * 1.1, pc_loadings_scaled[1, i] * 1.1, var, color='darkblue', fontsize=10)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_title('PCA Biplot')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Step 7: Heatmaps of Feature Contributions to PCs
    contributions_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=dates
    )

    def plot_pc_contribution_heatmap(pc_number, title):
        fig, ax = plt.subplots(figsize=(14, 24))
        formatted_dates = dates.dt.date
        heatmap_data = pd.DataFrame(
            np.outer(contributions_df[f'PC{pc_number}'], pca.components_[pc_number - 1]),
            index=formatted_dates,
            columns=df_features.columns
        )
        sns.heatmap(
            heatmap_data,
            cmap='bwr',
            cbar=True,
            vmin=-10,
            vmax=10,
            ax=ax
        )
        ax.set_title(f'{title} (PC{pc_number})')
        ax.set_xlabel('Features')
        ax.set_ylabel('Dates')
        st.pyplot(fig)

    st.subheader("Heatmaps of Feature Contributions")
    plot_pc_contribution_heatmap(1, 'Feature Contributions to Principal Component 1')
    plot_pc_contribution_heatmap(2, 'Feature Contributions to Principal Component 2')
    plot_pc_contribution_heatmap(3, 'Feature Contributions to Principal Component 3')


# Streamlit App
st.title("PCA Analysis App")

# Dropdown for selecting periods
periods = ['1d', '10d', '30d', '90d']
# Dropdown to select the period
selected_period = st.selectbox("Select the period:", periods)

# Dropdown to filter by Absolute or Relative returns
return_type = st.selectbox(
    "Absolute / Relative Returns:",
    options=["Absolute Returns (return_abs)", "Relative Returns (pct_change)"],
    index=0  # Default to "Absolute Returns"
)

# Determine the suffix to filter by based on selection
return_suffix = "_return_abs" if return_type == "Absolute Returns (return_abs)" else "_pct_change"

# Get columns for the selected period with the chosen return type
period_columns = [
    col for col in df.columns if f"_{selected_period}_" in col and col.endswith(return_suffix)
]

# Define categories for further filtering
feature_categories = {
    "SARON (Compound Rates)": [col for col in period_columns if "Compound Rate" in col],
    "SARON (Close of Trading Rates)": [col for col in period_columns if "close of trading" in col],
    "Government Bond Yields": [col for col in period_columns if "year" in col],
    "CHF EUR Exchange Rate": [col for col in period_columns if "CHFEUR" in col]
}

# Multiselect for feature category selection
selected_categories = st.multiselect(
    "Select feature categories to include:",
    list(feature_categories.keys()),
    default=list(feature_categories.keys())  # Default to all categories selected
)

# Filter features based on the selected categories
filtered_features = [
    feature for category in selected_categories for feature in feature_categories[category]
]

# Initialize session state for selected features if not already present
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

# Initialize session state for selected period and return type
if "selected_period" not in st.session_state:
    st.session_state.selected_period = periods[0]

if "return_type" not in st.session_state:
    st.session_state.return_type = "Absolute Returns (return_abs)"

if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

# Handle changes in selected period or return type
if selected_period != st.session_state.selected_period or return_type != st.session_state.return_type:
    st.session_state.selected_period = selected_period
    st.session_state.return_type = return_type
    st.session_state.selected_features = []  # Reset selected features if period or return type changes

# Determine the suffix to filter by based on selection
return_suffix = "_return_abs" if st.session_state.return_type == "Absolute Returns (return_abs)" else "_pct_change"

# Get columns for the selected period with the chosen return type
period_columns = [
    col for col in df.columns if f"_{st.session_state.selected_period}_" in col and col.endswith(return_suffix)
]

# Define categories for further filtering
feature_categories = {
    "SARON (Compound Rates)": [col for col in period_columns if "Compound Rate" in col],
    "SARON (Close of Trading Rates)": [col for col in period_columns if "close of trading" in col],
    "Government Bond Yields": [col for col in period_columns if "year" in col],
    "CHF EUR Exchange Rate": [col for col in period_columns if "CHFEUR" in col]
}



# Filter features based on the selected categories
filtered_features = [
    feature for category in selected_categories for feature in feature_categories[category]
]

# Validate session state selected features against the current filtered features
valid_selected_features = [
    feature for feature in st.session_state.selected_features if feature in filtered_features
]

# Update the session state if invalid features are removed
if valid_selected_features != st.session_state.selected_features:
    st.session_state.selected_features = valid_selected_features

# Multiselect for choosing specific features
selected_features = st.multiselect(
    "Select the features to include in PCA:",
    options=filtered_features,
    default=st.session_state.selected_features,  # Set default to valid selections in session state
    key="selected_features_multiselect"  # Assign a unique key
)

# Update session state only if the selection changes
if selected_features != st.session_state.selected_features:
    st.session_state.selected_features = selected_features

# Perform PCA analysis if features are selected
if st.session_state.selected_features:
    perform_pca_analysis_streamlit(df, st.session_state.selected_features)
else:
    st.error("Please select at least one feature for PCA analysis.")


