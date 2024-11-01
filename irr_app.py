# import streamlit as st
# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt

# # Constants for depth increment
# depth_increment = 12  # 12-inch increments assumed for each depth

# # Pre-set paths to files
# field_boundaries_file = "/Users/simdua/Library/CloudStorage/OneDrive-KansasStateUniversity/Documents/Documents KSU/Hackathone 2024/Datasets/Plot boundaries/Map with all plots/2024_Colby_TAPS_Harvest_Area.shp" 

# # Load field boundaries at start
# field_boundaries = gpd.read_file(field_boundaries_file)

# # Field capacity and wilting point values for each soil texture type
# texture_field_capacity = {
#     "Sand": 0.17,
#     "Loamy Sand": 0.09,
#     "Sandy Loam": 0.14,
#     "Loam": 0.25,
#     "Silt Loam": 0.28,
#     "Clay Loam": 0.32,
#     "Silty Clay Loam": 0.30,
#     "Clay": 0.32
# }

# texture_wilting_point = {
#     "Sand": 0.025,
#     "Loamy Sand": 0.04,
#     "Sandy Loam": 0.05,
#     "Loam": 0.12,
#     "Silt Loam": 0.09,
#     "Clay Loam": 0.15,
#     "Silty Clay Loam": 0.17,
#     "Clay": 0.20
# }

# # Streamlit app code
# st.title("Irrigation recoomenddation")

# # User file upload for Neutron Probe data
# neutron_data_file = st.file_uploader("Upload Neutron Probe Data Excel File", type=["xlsx"])

# if neutron_data_file:
#     neutron_data = pd.read_excel(neutron_data_file, skiprows=2)
#     neutron_data.rename(columns={'Plot #': 'Plot_ID'}, inplace=True)

#     # Inputs for field capacity and wilting point based on soil texture
#     st.sidebar.header("Soil Texture and Irrigation Settings")
#     texture_type = st.sidebar.selectbox("Select Soil Texture", ["Sand", "Loamy Sand", "Sandy Loam", "Loam", "Silt Loam", "Clay Loam", "Silty Clay Loam", "Clay"])

#     # Assign field capacity and wilting point from dictionaries based on selected texture
#     field_capacity = texture_field_capacity[texture_type]
#     wilting_point = texture_wilting_point[texture_type]

#     # st.sidebar.write(f"Field Capacity for {texture_type}: {field_capacity}")
#     # st.sidebar.write(f"Wilting Point for {texture_type}: {wilting_point}")

#     # Merge neutron data with field boundaries
#     merged_gdf = field_boundaries.merge(neutron_data, on='Plot_ID', how='inner')

#     # Calculate SWC and PAW
#     depths = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]
#     for depth in depths:
#         merged_gdf[f'{depth}_SWC'] = merged_gdf[depth] * (depth_increment / 12)
#         merged_gdf[f'{depth}_PAW'] = (merged_gdf[depth] - wilting_point) * (depth_increment / 12)
#         merged_gdf[f'{depth}_PAW'] = merged_gdf[f'{depth}_PAW'].clip(lower=0)

#     # Calculate Total PAW and Root Zone SWC
#     merged_gdf['Total_PAW'] = merged_gdf[[f'{depth}_PAW' for depth in depths]].sum(axis=1)
#     merged_gdf['Root_Zone_SWC'] = merged_gdf[[f'{depth}_SWC' for depth in depths]].sum(axis=1)

#     # Determine irrigation needs
#     max_paw = (field_capacity - wilting_point) * (len(depths) * depth_increment / 12)
#     merged_gdf['Irrigation_Needed'] = merged_gdf['Total_PAW'] < 0.5 * max_paw

#     # Display irrigation recommendations
#     st.subheader("Irrigation Recommendations")
#     # st.write(merged_gdf[['Plot_ID', 'Root_Zone_SWC', 'Total_PAW', 'Irrigation_Needed']])

#     # Visualization of soil moisture at selected depth
#     selected_depth = st.selectbox("Select Depth for Soil Moisture Visualization (inches)", depths)
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     merged_gdf.plot(column=f'{selected_depth}_SWC', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
#     ax.set_title(f'Soil Moisture at {selected_depth}-inch Depth')
#     st.pyplot(fig)

#     # Visualization: Irrigation Needs Map
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     merged_gdf.plot(column='Irrigation_Needed', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
#     ax.set_title('Irrigation Needs (Yes = 1, No = 0)')
#     st.pyplot(fig)



##################################
####################################

# import streamlit as st
# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt

# # Constants for depth increment
# depth_increment = 12  # assuming each depth represents a 12-inch increment

# field_boundaries_file = "/Users/simdua/Library/CloudStorage/OneDrive-KansasStateUniversity/Documents/Documents KSU/Hackathone 2024/Datasets/Plot boundaries/Map with all plots/2024_Colby_TAPS_Harvest_Area.shp"
# # Load field boundaries at start
# field_boundaries = gpd.read_file(field_boundaries_file)

# # Field capacity and wilting point values for each soil texture type
# texture_field_capacity = {
#     "Sand": 0.10,
#     "Loamy Sand": 0.12,
#     "Sandy Loam": 0.14,
#     "Loam": 0.20,
#     "Silt Loam": 0.25,
#     "Clay Loam": 0.30,
#     "Silty Clay Loam": 0.33,
#     "Clay": 0.38
# }

# texture_wilting_point = {
#     "Sand": 0.05,
#     "Loamy Sand": 0.06,
#     "Sandy Loam": 0.10,
#     "Loam": 0.12,
#     "Silt Loam": 0.15,
#     "Clay Loam": 0.18,
#     "Silty Clay Loam": 0.20,
#     "Clay": 0.25
# }

# # Crop Coefficients for different crop types
# crop_coefficients = {
#     "Corn": 1.15,
#     "Wheat": 1.2,
#     "Soybeans": 1.05,
#     "Rice": 1.1,
#     "Vegetables": 0.95,
#     "Orchard": 0.85
# }

# # Streamlit app code
# st.title("Irrigation Recommendation App")

# # User inputs for ET0 and crop type
# et0 = st.sidebar.number_input("Enter local reference ET0 (mm/day):", value=5.0)
# crop_type = st.sidebar.selectbox("Select Crop Type", list(crop_coefficients.keys()))
# soil_texture = st.sidebar.selectbox("Select Soil Texture", list(texture_field_capacity.keys()))

# # Retrieve corresponding field capacity, wilting point, and Kc value
# field_capacity = texture_field_capacity[soil_texture]
# wilting_point = texture_wilting_point[soil_texture]
# kc = crop_coefficients[crop_type]

# # Calculate ETc using the selected crop coefficient
# etc = et0 * kc
# st.sidebar.write(f"Calculated ETc (mm/day) for {crop_type}: {etc}")

# # User file upload for Neutron Probe data
# neutron_data_file = st.file_uploader("Upload Neutron Probe Data Excel File", type=["xlsx"])

# if neutron_data_file:
#     neutron_data = pd.read_excel(neutron_data_file, skiprows=2)
#     neutron_data.rename(columns={'Plot #': 'Plot_ID'}, inplace=True)

#     # Merge neutron data with field boundaries
#     merged_gdf = field_boundaries.merge(neutron_data, on='Plot_ID', how='inner')

#     # Calculate SWC and PAW
#     depths = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]
#     for depth in depths:
#         merged_gdf[f'{depth}_SWC'] = merged_gdf[depth] * (depth_increment / 12)
#         merged_gdf[f'{depth}_PAW'] = (merged_gdf[depth] - wilting_point) * (depth_increment / 12)
#         merged_gdf[f'{depth}_PAW'] = merged_gdf[f'{depth}_PAW'].clip(lower=0)

#     # Calculate Total PAW and Root Zone SWC
#     merged_gdf['Total_PAW'] = merged_gdf[[f'{depth}_PAW' for depth in depths]].sum(axis=1)
#     merged_gdf['Root_Zone_SWC'] = merged_gdf[[f'{depth}_SWC' for depth in depths]].sum(axis=1)
#     ############
#     max_PAW = (field_capacity - wilting_point) * (sum([depth_increment/12 for depth in depths]))
#     #############
#     # Determine irrigation needs based on ETc
#     merged_gdf['Irrigation_Needed'] = merged_gdf['Total_PAW'] < etc  # Use ETc as threshold

#     ###########
#     # Calculate irrigation requirement
#     merged_gdf['MAD_threshold'] = max_PAW * (1 - management_allowed_depletion)
#     merged_gdf['Irrigation_Required'] = (merged_gdf['MAD_threshold'] -merged_gdf['Total_PAW']).clip(lower=0)


    
#     # Display irrigation recommendations
#     st.subheader("Irrigation Recommendations")
#     st.write(merged_gdf[['Plot_ID', 'Root_Zone_SWC', 'Total_PAW', 'Irrigation_Needed','Irrigation_Required']])

#     # Visualization of soil moisture at selected depth
#     selected_depth = st.selectbox("Select Depth for Soil Moisture Visualization (inches)", depths)
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     merged_gdf.plot(column=f'{selected_depth}_SWC', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
#     ax.set_title(f'Soil Moisture at {selected_depth}-inch Depth')
#     st.pyplot(fig)

#     # Visualization: Irrigation Needs Map
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     merged_gdf.plot(column='Irrigation_Needed', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
#     ax.set_title('Irrigation Needs (Yes = 1, No = 0)')
#     st.pyplot(fig)

#  #########
#      #Create visualizations
#     # 1. Irrigation Requirements Map
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#     merged_gdf.plot(column='Irrigation_Required', 
#                     cmap='RdYlBu_r', 
#                     legend=True,
#                     legend_kwds={'label': 'Irrigation Required (inches)'},
#                     ax=ax)
#     ax.set_title(f'Irrigation Requirements Map\
#     Crop: {crop_type}, Soil: {soil_texture}')
#     plt.show()

#         print("\
#     Field Statistics:")
#     summary_stats = merged_gdf[['Irrigation_Required', 'Total_PAW', 'Root_Zone_SWC']].describe().round(2)
#     print(summary_stats)
#     # Create irrigation priority categories
#     def get_priority(row):
#         if row['Irrigation_Required'] > 1.0:
#             return 'High Priority'
#         elif row['Irrigation_Required'] > 0.5:
#             return 'Medium Priority'
#         else:
#             return 'Low Priority'

#     merged_gdf['Irrigation_Priority'] = merged_gdf.apply(get_priority, axis=1)

#     # Print irrigation priorities
#     print("\
#     Irrigation Priorities by Plot:")
#     priority_summary = merged_gdf.groupby('Irrigation_Priority')['Plot_ID'].count()
#     print(priority_summary)

#     # Save results to CSV
#     output_filename = f'irrigation_recommendations_{crop_type}_{soil_texture}.csv'
#     merged_gdf[['Plot_ID', 'Total_PAW', 'Root_Zone_SWC', 'Irrigation_Required', 'Irrigation_Priority']].to_csv(output_filename, index=False)
#     print(f"\
#     Detailed recommendations saved to: {output_filename}")
# #######################################################
######################################################


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import geopandas as gpd

# # Constants from the streamlit app
# texture_field_capacity = {
#     "Sand": 0.10,
#     "Loamy Sand": 0.12,
#     "Sandy Loam": 0.14,
#     "Loam": 0.20,
#     "Silt Loam": 0.25,
#     "Clay Loam": 0.30,
#     "Silty Clay Loam": 0.33,
#     "Clay": 0.38
# }

# texture_wilting_point = {
#     "Sand": 0.05,
#     "Loamy Sand": 0.06,
#     "Sandy Loam": 0.10,
#     "Loam": 0.12,
#     "Silt Loam": 0.15,
#     "Clay Loam": 0.18,
#     "Silty Clay Loam": 0.20,
#     "Clay": 0.25
# }

# crop_coefficients = {
#     "Corn": 1.15,
#     "Wheat": 1.2,
#     "Soybeans": 1.05,
#     "Rice": 1.1,
#     "Vegetables": 0.95,
#     "Orchard": 0.85
# }

# # Load the data
# df = pd.read_excel('24 KSU TAPS Neutron Tube Readings_VWC.xlsx', skiprows=2, engine='calamine')
# depth_columns = ['Date', 'Plot #', 'Block #'] + [str(x) for x in [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]]
# df.columns = depth_columns

# # Load field boundaries
# field_boundaries = gpd.read_file('2024_Colby_TAPS_Harvest_Area.shp')

# # Example parameters (these would be user inputs in Streamlit)
# et0 = 5.0  # mm/day
# soil_texture = "Loam"  # example soil type
# crop_type = "Corn"    # example crop type

# # Get soil parameters based on texture
# field_capacity = texture_field_capacity[soil_texture]
# wilting_point = texture_wilting_point[soil_texture]
# kc = crop_coefficients[crop_type]

# # Calculate ETc
# etc = et0 * kc

# # Constants
# depth_increment = 12    # inches between measurements
# management_allowed_depletion = 0.50  # MAD (50%)

# # Rename Plot # to Plot_ID for merging
# df = df.rename(columns={'Plot #': 'Plot_ID'})

# # Merge with field boundaries
# merged_gdf = field_boundaries.merge(df, on='Plot_ID', how='inner')

# depths = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]

# # Calculate SWC and PAW for each depth
# for depth in depths:
#     # Convert VWC to inches of water
#     merged_gdf[f'{depth}_SWC'] = merged_gdf[str(depth)] * (depth_increment / 12)
    
#     # Calculate PAW
#     merged_gdf[f'{depth}_PAW'] = (merged_gdf[str(depth)] - wilting_point) * (depth_increment / 12)
#     merged_gdf[f'{depth}_PAW'] = merged_gdf[f'{depth}_PAW'].clip(lower=0)

# # Calculate total PAW and maximum PAW
# merged_gdf['Total_PAW'] = merged_gdf[[f'{depth}_PAW' for depth in depths]].sum(axis=1)
# max_PAW = (field_capacity - wilting_point) * (sum([depth_increment/12 for depth in depths]))

# # Calculate Root Zone SWC
# merged_gdf['Root_Zone_SWC'] = merged_gdf[[f'{depth}_SWC' for depth in depths]].sum(axis=1)

# # Calculate irrigation requirement
# merged_gdf['MAD_threshold'] = max_PAW * (1 - management_allowed_depletion)
# merged_gdf['Irrigation_Required'] = (merged_gdf['MAD_threshold'] - merged_gdf['Total_PAW']).clip(lower=0)

# # Create visualizations
# # 1. Irrigation Requirements Map
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# merged_gdf.plot(column='Irrigation_Required', 
#                 cmap='RdYlBu_r', 
#                 legend=True,
#                 legend_kwds={'label': 'Irrigation Required (inches)'},
#                 ax=ax)
# ax.set_title(f'Irrigation Requirements Map\
# Crop: {crop_type}, Soil: {soil_texture}')
# plt.show()

# # 2. Soil Moisture Profile
# selected_depth = 30  # example depth
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# merged_gdf.plot(column=f'{selected_depth}_SWC',
#                 cmap='Blues',
#                 legend=True,
#                 legend_kwds={'label': f'Soil Water Content at {selected_depth} inches'},
#                 ax=ax)
# ax.set_title(f'Soil Moisture at {selected_depth}-inch Depth')
# plt.show()

# # Print summary statistics
# print(f"\
# Irrigation Summary for {crop_type} on {soil_texture} soil:")
# print("\
# Field Statistics:")
# summary_stats = merged_gdf[['Irrigation_Required', 'Total_PAW', 'Root_Zone_SWC']].describe().round(2)
# print(summary_stats)

# # Create irrigation priority categories
# def get_priority(row):
#     if row['Irrigation_Required'] > 1.0:
#         return 'High Priority'
#     elif row['Irrigation_Required'] > 0.5:
#         return 'Medium Priority'
#     else:
#         return 'Low Priority'

# merged_gdf['Irrigation_Priority'] = merged_gdf.apply(get_priority, axis=1)

# # Print irrigation priorities
# print("\
# Irrigation Priorities by Plot:")
# priority_summary = merged_gdf.groupby('Irrigation_Priority')['Plot_ID'].count()
# print(priority_summary)

# # Save results to CSV
# output_filename = f'irrigation_recommendations_{crop_type}_{soil_texture}.csv'
# merged_gdf[['Plot_ID', 'Total_PAW', 'Root_Zone_SWC', 'Irrigation_Required', 'Irrigation_Priority']].to_csv(output_filename, index=False)
# print(f"\
# Detailed recommendations saved to: {output_filename}")






##########################################################
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Constants for depth increment
depth_increment = 12  # assuming each depth represents a 12-inch increment
management_allowed_depletion = 0.50  # MAD (50%)

field_boundaries_file = "/Users/simdua/Library/CloudStorage/OneDrive-KansasStateUniversity/Documents/Documents KSU/Hackathone 2024/Datasets/Plot boundaries/Map with all plots/2024_Colby_TAPS_Harvest_Area.shp"
field_boundaries = gpd.read_file(field_boundaries_file)

# field capacityand wilting point values for each soil texture type
texture_field_capacity = {
    "Sand": 0.10, "Loamy Sand": 0.12, "Sandy Loam": 0.14, "Loam": 0.20, "Silt Loam": 0.25,
    "Clay Loam": 0.30, "Silty Clay Loam": 0.33, "Clay": 0.38
}
texture_wilting_point = {
    "Sand": 0.05, "Loamy Sand": 0.06, "Sandy Loam": 0.10, "Loam": 0.12,
    "Silt Loam": 0.15, "Clay Loam": 0.18, "Silty Clay Loam": 0.20, "Clay": 0.25
}

# Crop Coefficients for different crop types
crop_coefficients = {
    "Corn": 1.15, "Wheat": 1.2, "Soybeans": 1.05, "Rice": 1.1,
    "Vegetables": 0.95, "Orchard": 0.85
}

st.title("Irrigation Recommendation App")
et0 = st.sidebar.number_input("Enter local reference ET0 (mm/day):", value=5.0)
crop_type = st.sidebar.selectbox("Select Crop Type", list(crop_coefficients.keys()))
soil_texture = st.sidebar.selectbox("Select Soil Texture", list(texture_field_capacity.keys()))

field_capacity = texture_field_capacity[soil_texture]
wilting_point = texture_wilting_point[soil_texture]
kc = crop_coefficients[crop_type]
etc =et0 * kc
st.sidebar.write(f"Calculated ETc (mm/day) for {crop_type}: {etc}")
########

######
neutron_data_file = st.file_uploader("Upload Neutron Probe Data Excel File", type=["xlsx"])
if neutron_data_file:
    neutron_data = pd.read_excel(neutron_data_file, skiprows=2)
    neutron_data.rename(columns={'Plot #': 'Plot_ID'}, inplace=True) 
    ####
    # Date range selector
    min_date = neutron_data['Date'].min()
    max_date = neutron_data['Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Filter for specific plot and date range
    mask = (
            #(neutron_data['Plot #'] == Plot_ID) & 
            (neutron_data['Date'] >= pd.Timestamp(date_range[0])) & 
            (neutron_data['Date'] <= pd.Timestamp(date_range[1]))
        )
    plot_data = neutron_data[mask].copy()
    #####
    merged_gdf= plot_data.merge(plot_data,  on='Plot_ID', how='inner')
    merged_gdf = field_boundaries.merge(neutron_data, on='Plot_ID', how='inner')
    depths = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]
    for depth in depths:
        merged_gdf[f'{depth}_SWC'] = merged_gdf[(depth)] * (depth_increment / 12)
        merged_gdf[f'{depth}_PAW'] = (merged_gdf[(depth)] - wilting_point) * (depth_increment / 12)
        merged_gdf[f'{depth}_PAW'] = merged_gdf[f'{depth}_PAW'].clip(lower=0)

    merged_gdf['Total_PAW'] = merged_gdf[[f'{depth}_PAW' for depth in depths]].sum(axis=1)
    merged_gdf['Root_Zone_SWC'] = merged_gdf[[f'{depth}_SWC' for depth in depths]].sum(axis=1)
    max_PAW = (field_capacity - wilting_point) * (sum([depth_increment / 12 for depth in depths]))
    merged_gdf['Irrigation_Needed'] = merged_gdf['Total_PAW'] < etc

    merged_gdf['MAD_threshold'] = max_PAW * (1 - management_allowed_depletion)
    merged_gdf['Irrigation_Required'] = (merged_gdf['MAD_threshold'] - merged_gdf['Total_PAW']).clip(lower=0)

    st.subheader("Irrigation Recommendations")
    st.write(merged_gdf[['Plot_ID', 'Root_Zone_SWC', 'Total_PAW', 'Irrigation_Needed', 'Irrigation_Required']])
    selected_depth = st.selectbox("Select Depth for Soil Moisture Visualization (inches)", depths)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_gdf.plot(column=f'{selected_depth}_SWC', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title(f'Soil Moisture at {selected_depth}-inch Depth')
    st.pyplot(fig)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # merged_gdf.plot(column='Irrigation_Needed', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    # ax.set_title('Irrigation Needs (Yes = 1, No = 0)')
    # st.pyplot(fig)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged_gdf.plot(column='Irrigation_Required', cmap='RdYlBu_r', legend=True, legend_kwds={'label': 'Irrigation Required (inches)'}, ax=ax)
    ax.set_title(f'Irrigation Requirements Map\nCrop: {crop_type}, Soil: {soil_texture}')
    st.pyplot(fig)

    st.write("Field Statistics:")
    summary_stats = merged_gdf[['Irrigation_Required', 'Total_PAW', 'Root_Zone_SWC']].describe().round(2)
    st.write(summary_stats)

    def get_priority(row):
        if row['Irrigation_Required'] > 1.0:
            return 'High Priority'
        elif row['Irrigation_Required'] > 0.5:
            return 'Medium Priority'
        return 'Low Priority'

    merged_gdf['Irrigation_Priority'] = merged_gdf.apply(get_priority, axis=1)
    st.write("Irrigation Priorities by Plot:")
    priority_summary = merged_gdf.groupby('Irrigation_Priority')['Plot_ID'].count()
    st.write(priority_summary)

    
#############

        

        # Soil moisture profile for the most recent date
    st.subheader('Current Soil Moisture Profile')
    latest_date = merged_gdf['Date'].max()
    latest_data = merged_gdf[merged_gdf['Date'] == latest_date]
    fig_profile, ax_profile = plt.subplots(figsize=(10, 6))
    ax_profile.plot(depths, latest_data[depths].iloc[0], 'b-o', label='Soil Moisture')      
    ax_profile.axhline(y=wilting_point, color='r', linestyle='--', label='Wilting Point')
    ax_profile.axhline(y=field_capacity, color='g', linestyle='--', label='Field Capacity')
    ax_profile.set_xlabel('Depth (inches)')
    ax_profile.set_ylabel('Volumetric Water Content')
    ax_profile.legend()
    ax_profile.grid(True)
    plt.tight_layout()
    st.pyplot(fig_profile)
        
        # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Current Total PAW (inches)', 
                     f"{merged_gdf['Total_PAW'].iloc[-1]:.2f}")
    with col2:
        st.metric('Current Root Zone SWC (inches)', 
                     f"{merged_gdf['Root_Zone_SWC'].iloc[-1]:.2f}")
    with col3:
        st.metric('Required Irrigation (inches)', 
                     f"{merged_gdf['Irrigation_Required'].iloc[-1]:.2f}")
    with col4:
        st.metric('Days in Analysis', f"{len(plot_data)}")
        
        # Trend Analysis
    st.subheader('Trend Analysis')
    col1, col2 = st.columns(2)
        
    with col1:
            # Calculate daily change in PAW
        merged_gdf['PAW_Change'] =merged_gdf['Total_PAW'].diff()
            
            # Plot daily PAW change
        fig_change, ax_change = plt.subplots(figsize=(10, 6))
        ax_change.bar(merged_gdf['Date'], merged_gdf['PAW_Change'], 
                     color=['red' if x < 0 else 'green' for x in merged_gdf['PAW_Change']])
        ax_change.set_xlabel('Date')
        ax_change.set_ylabel('Daily Change in PAW (inches)')
        ax_change.set_title('Daily Change in Plant Available Water')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_change)
        
    with col2:
            # Depth-wise trend
        st.write('Soil Moisture Trends by Depth')
        trend_data = merged_gdf[depths].mean().round(3)
        trend_df = pd.DataFrame({
                'Depth (inches)': depths,
                'Average VWC': trend_data.values
            })
        st.dataframe(trend_df)
        
        # Recommendations based on trends
    st.subheader('Irrigation Recommendations')
        
        # Calculate average daily water loss
    # Assuming merged_gdf is properly set up earlier in your script
    avg_daily_loss = -merged_gdf['PAW_Change'][merged_gdf['PAW_Change'] < 0].mean()
    days_until_mad = ((merged_gdf['Total_PAW'].iloc[-1] - merged_gdf['MAD_threshold'].iloc[-1]) / avg_daily_loss if avg_daily_loss > 0 else float('inf'))

    if merged_gdf['Irrigation_Required'].iloc[-1] > 0:
        st.warning(f"Immediate irrigation of {merged_gdf['Irrigation_Required'].iloc[-1]:.2f} inches recommended")
    elif days_until_mad < 7:
        st.warning(f"Irrigation will be needed in approximately {days_until_mad:.1f} days")
    else:
        st.success(" No irrigation required")
###############
    
    
    
    
    
#     output_filename = f'irrigation_recommendations_{crop_type}_{soil_texture}.csv'
#     merged_gdf[['Plot_ID', 'Total_PAW', 'Root_Zone_SWC', 'Irrigation_Required', 'Irrigation_Priority']].to_csv(output_filename, index=False)
#     st.write(f"Detailed recommendations saved to: {output_filename}")
