# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np

# Code for the information related to the app
linkedin = "https://www.linkedin.com/in/jaymin-mistry-765902212"
st.header('Ranking Mobile Phones based on User Requests', divider='red')
st.caption(f'This app was built as a part of my TCS iON Internship Project. It aims to rank the mobile phones based on the interests of the user.\
           To use the app just adjust the sliders on the sidebar, select the connectivity options you want and click the button below. \
           The app will rank the 10 best phones according to your inputs. Connect with me on LinkedIn [here]({linkedin})')

st.sidebar.header('Adjust Feature Importance Scores', divider='red')
st.sidebar.caption('Adjust the scores according to your liking. Some things to remember are:')
st.sidebar.caption('1. These are the importance values and not the actual values of the features ie. higher value means the \
                   corresponding feature will be given more importance while calculating the final rankings. Zero value means that feature will \
                   not be considered for calculations.')
st.sidebar.caption('2. Do not set all scores as zero.')
st.sidebar.caption('3. The calculations are affected by relative scores ie. if you give all features importance of 10 it will be same as \
                   giving all features importance of 1. So only increase the score of features which you want.')
st.sidebar.caption('4. Features whose lower values are preferred like price are already accounted for in the calculation. Giving higher score \
                   to these features will not show higher values.')


# Stores features weights
weights = {
    'battery_power': 0,
    'clock_speed': 0,
    'fc': 0,
    'int_memory': 0,
    'n_cores': 0,
    'pc': 0,
    'ram': 0,
    'talk_time': 0,
    'px_area': 0,
    'sc_area': 0,
    'm_dep': 0,
    'mobile_wt': 0,
    'price_range': 0
}

# Code for the weight silders
weight = st.sidebar.slider(f'Score for Battery Capacity', 0, 10, 5)
weights['battery_power'] = weight

weight = st.sidebar.slider(f'Score for Clock Speed', 0, 10, 5)
weights['clock_speed'] = weight

weight = st.sidebar.slider(f'Score for Front Camera', 0, 10, 5)
weights['fc'] = weight

weight = st.sidebar.slider(f'Score for Internal Memory', 0, 10, 5)
weights['int_memory'] = weight

weight = st.sidebar.slider(f'Score for Mobile Depth', 0, 10, 5)
weights['m_dep'] = weight

weight = st.sidebar.slider(f'Score for Mobile Weight', 0, 10, 5)
weights['mobile_wt'] = weight

weight = st.sidebar.slider(f'Score for Number of cores', 0, 10, 5)
weights['n_cores'] = weight

weight = st.sidebar.slider(f'Score for Primary Camera', 0, 10, 5)
weights['pc'] = weight

weight = st.sidebar.slider(f'Score for RAM', 0, 10, 5)
weights['ram'] = weight

weight = st.sidebar.slider(f'Score for Talk Time', 0, 10, 5)
weights['talk_time'] = weight

weight = st.sidebar.slider(f'Score for Pixel Dots', 0, 10, 5)
weights['px_area'] = weight

weight = st.sidebar.slider(f'Score for Screen Area', 0, 10, 5)
weights['sc_area'] = weight

weight = st.sidebar.slider(f'Score for Price Range', 0, 10, 5)
weights['price_range'] = weight


# Beneficial variables where higher values are preferred
benf = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'ram', 'talk_time', 'px_area', 'sc_area']

# Non-beneficial variables where lower values are preferred
non_benf = ['m_dep', 'mobile_wt', 'price_range']

# Function for loading data
@st.cache_data(ttl=3600, max_entries=100, show_spinner="Fetching data...")
def load_data(url):
    df = pd.read_csv(url)
    return df

# Code for the Connectivity Options button
st.subheader('Connectivity Options')
bluetooth_option = st.radio('Bluetooth', ('Yes', 'No', 'Any'), index=2, horizontal=True)
dual_sim_option = st.radio('Dual Sim', ('Yes', 'No', 'Any'), index=2, horizontal=True)
g4_option = st.radio('4G', ('Yes', 'No', 'Any'), index=2, horizontal=True)
g3_option = st.radio('3G', ('Yes', 'No', 'Any'), index=2, horizontal=True)
touch_option = st.radio('Touch Screen', ('Yes', 'No', 'Any'), index=2, horizontal=True)
wifi_option = st.radio('Wifi', ('Yes', 'No', 'Any'), index=2, horizontal=True)

# Function to filter data
def filter_data(df):
    data = df.copy()
    
    # Filter based on Bluetooth option
    if bluetooth_option == 'Yes':
        data = data[data['blue'] == 1]
    elif bluetooth_option == 'No':
        data = data[data['blue'] == 0]
    
    # Filter based on Dual Sim option
    if dual_sim_option == 'Yes':
        data = data[data['dual_sim'] == 1]
    elif dual_sim_option == 'No':
        data = data[data['dual_sim'] == 0]
    
    # Filter based on 4G option
    if g4_option == 'Yes':
        data = data[data['four_g'] == 1]
    elif g4_option == 'No':
        data = data[data['four_g'] == 0]
    
    # Filter based on 3G option
    if g3_option == 'Yes':
        data = data[data['three_g'] == 1]
    elif g3_option == 'No':
        data = data[data['three_g'] == 0]
    
    # Filter based on Touch Screen option
    if touch_option == 'Yes':
        data = data[data['touch_screen'] == 1]
    elif touch_option == 'No':
        data = data[data['touch_screen'] == 0]
    
    # Filter based on Wifi option
    if wifi_option == 'Yes':
        data = data[data['wifi'] == 1]
    elif wifi_option == 'No':
        data = data[data['wifi'] == 0]
    
    return data

# Function to normalize data
def normalize_data(df):
    norm_df = pd.DataFrame()
    normalized_benf = df[benf].apply(lambda x: x / np.linalg.norm(x), axis=0)
    norm_df[benf] = normalized_benf

    normalized_non_benf = df[non_benf].apply(lambda x: x / np.linalg.norm(x), axis=0)
    norm_df[non_benf] = normalized_non_benf
    return norm_df

# Function to calculate the performance scores
def calc_scores(weights):
    try:
        total_weight = sum(weights.values())
        normalized_weights = {col: weight / total_weight for col, weight in weights.items()}

        # Create the weighted df
        weighted_df = normalized_df.copy()
        for col in normalized_weights:
            weighted_df[col] = weighted_df[col] * normalized_weights[col]

        # Calculate distance from best solution
        best_df = pd.DataFrame()
        for col in weighted_df.columns:
            if col in benf:
                best_df[col] = (weighted_df[col] - max(weighted_df[col]))**2
            else:
                best_df[col] = (weighted_df[col] - min(weighted_df[col]))**2
        best_df['score'] = np.sqrt(best_df.sum(axis=1))

        # Calculate distance from worst solution
        worst_df = pd.DataFrame()
        for col in weighted_df.columns:
            if col in benf:
                worst_df[col] = (weighted_df[col] - min(weighted_df[col]))**2
            else:
                worst_df[col] = (weighted_df[col] - max(weighted_df[col]))**2
        worst_df['score'] = np.sqrt(worst_df.sum(axis=1))

        # Calculate the performance score
        temp = final_df.iloc[normalized_df.index,:].copy()
        temp['Performance Score'] = worst_df['score'] / (best_df['score'] + worst_df['score'])
        return temp
    except:
        st.error("Please adjust your weights to avoid zero weights for all features.")
        return None

# Function to rank the final dataset
def get_ranks(df):
    df['Rank'] = df['Performance Score'].rank(ascending=False, method='dense')
    df.sort_values(by='Rank', inplace=True)
    df = df[['Rank'] + [col for col in df.columns if col != 'Rank']]
    df.set_index('Rank', inplace=True)
    return df

# Code for the button to run
if st.button("Calculate Rankings", type='primary'):
    mobile_df = load_data("https://raw.githubusercontent.com/Jaymin151617/TCS-RIO-125-Internship/main/mobile.csv")
    final_df = load_data("https://raw.githubusercontent.com/Jaymin151617/TCS-RIO-125-Internship/main/final_mobile.csv")
    filter_df = filter_data(mobile_df)
    normalized_df = normalize_data(filter_df)
    calc_df = calc_scores(weights)
    if calc_df is not None:
        rank_df = get_ranks(calc_df)
        st.dataframe(rank_df.head(10))
