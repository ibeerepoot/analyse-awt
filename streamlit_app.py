import numpy as np
import pandas as pd
import streamlit as st
import json
from io import StringIO
from datetime import datetime, timedelta
import re
import csv
import altair as alt
import zipfile

"""
# Let's analyse your AWT data!
"""

# Sidebar for accepting input parameters
with st.sidebar:
    # Load AWT data
    st.header('Upload your data')
    st.markdown('**1. AWT data**')
    awt_uploaded_file = st.file_uploader("Upload your Tockler data here. You can export your data by going to Tockler > Search > Set a time period > Export to CSV.")

    # Load Google Maps data
    st.markdown('**2. Google Maps Timeline**')
    google_maps_uploaded_files = st.file_uploader("Upload your Google Maps Timeline data here. You can export your data by going to https://myaccount.google.com/yourdata/maps?hl=en > Download your Maps data > Check Location History (Timeline). There should be one JSON file per month.", type='json', accept_multiple_files=True)

    # Load Survey results data
    st.markdown('**3. Survey results**')
    survey_uploaded_file = st.file_uploader("Upload your survey results here. The CSV should contain 5 columns: Date, Productivity, Vigor, Dedication, Absorption.")

    # Load stress data
    st.markdown('**4. Stress data**')
    stress_data_zip = st.file_uploader("Upload your stress data (ZIP file) here.", type='zip')

    # Load heart rate data
    st.markdown('**5. Heart rate data**')
    heart_rate_data_zip = st.file_uploader("Upload your heart rate data (ZIP file) here.", type='zip')

# Main section for processing AWT data
if awt_uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a dataframe
        awt_stringio = StringIO(awt_uploaded_file.getvalue().decode('latin1'))
        
        # Explicitly set the delimiter as semicolon
        dataframe_awt = pd.read_csv(awt_stringio, delimiter=';')

        # Drop the 'Type' column if it exists
        if 'Type' in dataframe_awt.columns:
            dataframe_awt = dataframe_awt.drop(columns=['Type'])

        # Display the first 5 rows of the dataframe
        # st.write("Snippet of the raw AWT data:")
        # st.write(dataframe_awt.head())

        # Remove rows where 'Begin' is empty
        dataframe_awt = dataframe_awt.dropna(subset=['Begin'])
        dataframe_awt = dataframe_awt[dataframe_awt['Begin'] != '']

        # Remove rows where 'Title' is 'NO_TITLE'
        dataframe_awt = dataframe_awt[dataframe_awt['Title'] != 'NO_TITLE']

        # Initialize lists to store merged rows
        merged_rows = []

        # Convert 'App' column to string
        dataframe_awt['App'] = dataframe_awt['App'].astype(str)

        # Convert 'Title' column to string
        dataframe_awt['Title'] = dataframe_awt['Title'].astype(str)

        # Iterate over the DataFrame to merge consecutive rows
        current_row = None
        for index, row in dataframe_awt.iterrows():
            if current_row is None:
                current_row = row
            else:
                # Check if the current row is consecutive with the previous row
                if row['Begin'] == current_row['End']:
                    # Merge titles and update End time
                    current_row['App'] += '; ' + row['App']
                    current_row['Title'] += '; ' + row['Title']
                    current_row['End'] = row['End']
                else:
                    # Append the current merged row to the list
                    merged_rows.append(current_row)
                    # Start a new merged row
                    current_row = row

        # Append the last merged row
        if current_row is not None:
            merged_rows.append(current_row)

        # Create a new DataFrame with the merged rows
        dataframe_merged_awt = pd.DataFrame(merged_rows)

        # Filter out rows with unwanted titles
        dataframe_merged_awt = dataframe_merged_awt[~dataframe_merged_awt['Title'].isin(['NO_TITLE', 'Windows Default Lock Screen'])]

        # Reset the index of the new DataFrame
        dataframe_merged_awt.reset_index(drop=True, inplace=True)

        st.write("AWT data merged to continued work slots:")
        st.write(dataframe_merged_awt.head())

    except pd.errors.ParserError as e:
        st.error(f"Error parsing AWT CSV file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Function to convert timestamp format and adjust timezone
def format_timestamp(timestamp):
    # Regular expression to match timestamp with or without milliseconds
    match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?Z", timestamp)
    if match:
        base_time = match.group(1)
        dt = datetime.strptime(base_time, "%Y-%m-%dT%H:%M:%S")
        dt = dt + timedelta(hours=2)  # Adjusting to UTC+2 for Netherlands
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

# Function to process stress data from a ZIP file and return a DataFrame
def process_stress_data_from_zip(zip_file, data_type):
    dataframes = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.json'):
                with zip_ref.open(file_info) as file:
                    json_content = json.load(file)
                    df = pd.DataFrame(json_content)
                    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
                    df['end_time'] = pd.to_datetime(df['end_time'], unit='ms')
                    df['Date'] = df['start_time'].dt.date
                    df['hour'] = df['start_time'].dt.hour  # Extract hour of day
                    df['day_of_week'] = df['start_time'].dt.day_name()  # Extract day of week
                    df['week_in_year'] = df['start_time'].dt.strftime('%U').astype(int)  # Week number in year
                    dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    if 'score' in combined_df.columns:
        combined_df['score'] = pd.to_numeric(combined_df['score'], errors='coerce')
    return combined_df

# Function to process heart rate data from a ZIP file and return a DataFrame
def process_heart_rate_data_from_zip(zip_file, data_type):
    dataframes = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.json'):
                with zip_ref.open(file_info) as file:
                    json_content = json.load(file)
                    df = pd.DataFrame(json_content)
                    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
                    df['end_time'] = pd.to_datetime(df['end_time'], unit='ms')
                    df['Date'] = df['start_time'].dt.date
                    df['hour'] = df['start_time'].dt.hour  # Extract hour of day
                    df['day_of_week'] = df['start_time'].dt.day_name()  # Extract day of week
                    df['week_in_year'] = df['start_time'].dt.strftime('%U').astype(int)  # Week number in year
                    dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    if 'heart_rate' in combined_df.columns:
        combined_df['heart_rate'] = pd.to_numeric(combined_df['heart_rate'], errors='coerce')
    return combined_df

# Check if Google Maps files have been uploaded 
if google_maps_uploaded_files:
    place_visits = []

    # Iterate over each uploaded file
    for uploaded_file in google_maps_uploaded_files:
        # Read the JSON file
        maps_data = json.load(uploaded_file)

        # Iterate over each entry in the JSON data
        for entry in maps_data.get("timelineObjects", []):
            if "placeVisit" in entry:
                place_visit = entry["placeVisit"]
                location = place_visit.get("location", {})
                duration = place_visit.get("duration", {})

                # Extract the required information
                place_visited = location.get("address", "Unknown")
                start_visit = duration.get("startTimestamp", "Unknown")
                end_visit = duration.get("endTimestamp", "Unknown")

                # Format the timestamps
                if start_visit != "Unknown":
                    start_visit = format_timestamp(start_visit)
                if end_visit != "Unknown":
                    end_visit = format_timestamp(end_visit)

                # Append the data as a dictionary to the list
                place_visits.append({
                    "Place_visited": place_visited,
                    "Start_visit": start_visit,
                    "End_visit": end_visit
                })

    # Create a DataFrame from the list of place visits
    dataframe_locations = pd.DataFrame(place_visits, columns=["Place_visited", "Start_visit", "End_visit"])

    # Display the DataFrame in Streamlit
    # st.write("Google Maps locations visited:")
    # st.write(dataframe_locations)

    # Initialize a list to store the enriched rows
    enriched_rows = []

    # Iterate over the rows in dataframe_merged_awt
    for index, row in dataframe_merged_awt.iterrows():
        begin_time = datetime.strptime(row['Begin'], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(row['End'], "%Y-%m-%d %H:%M:%S")
        location = "Unknown"

        # Check for corresponding location in dataframe_locations
        for loc_index, loc_row in dataframe_locations.iterrows():
            loc_start = datetime.strptime(loc_row['Start_visit'], "%Y-%m-%d %H:%M:%S")
            loc_end = datetime.strptime(loc_row['End_visit'], "%Y-%m-%d %H:%M:%S")
            if loc_start <= begin_time <= loc_end or loc_start <= end_time <= loc_end or (begin_time <= loc_start and end_time >= loc_end):
                location = loc_row['Place_visited']
                break

        # Append the enriched data to the list
        enriched_rows.append({
            "App": row['App'],
            "Title": row['Title'],
            "Begin": begin_time,  # Convert to datetime object
            "End": end_time,      # Convert to datetime object
            "Location": location
        })

    # Create a new DataFrame with the enriched rows
    dataframe_merged_awt_with_locations = pd.DataFrame(enriched_rows)

    # Display the enriched DataFrame in Streamlit
    # st.write("Enriched AWT data with locations:")
    # st.write(dataframe_merged_awt_with_locations)

# Check if a Survey results file has been uploaded
if survey_uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a dataframe
        survey_stringio = StringIO(survey_uploaded_file.getvalue().decode('utf-8'))
        dialect = csv.Sniffer().sniff(survey_stringio.read(1024))
        survey_stringio.seek(0)
        dataframe_survey = pd.read_csv(survey_stringio, delimiter=dialect.delimiter)

        # Display the first 5 rows of the dataframe
        # st.write("Snippet of the survey results data:")
        # st.write(dataframe_survey.head())

        # Convert survey date format to match
        dataframe_survey['Date'] = pd.to_datetime(dataframe_survey['Date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

    except pd.errors.ParserError as e:
        st.error(f"Error parsing Survey CSV file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Process stress data if uploaded
if stress_data_zip:
    stress_data_df = process_stress_data_from_zip(stress_data_zip, 'score')
    if 'hour' not in stress_data_df.columns:
        st.error("Hour column missing in stress data.")
    else:
        # Calculate average stress score per hour
        average_stress_per_hour = stress_data_df.groupby('hour').agg(
            average_value=('score', 'mean')
        ).reset_index()
        average_stress_per_hour['data_type'] = 'Stress'  # Add data type for legend

# Process heart rate data if uploaded
if heart_rate_data_zip:
    heart_rate_data_df = process_heart_rate_data_from_zip(heart_rate_data_zip, 'heart_rate')
    if 'hour' not in heart_rate_data_df.columns:
        st.error("Hour column missing in heart rate data.")
    else:
        # Calculate average heart rate per hour
        average_heart_rate_per_hour = heart_rate_data_df.groupby('hour').agg(
            average_value=('heart_rate', 'mean')
        ).reset_index()
        average_heart_rate_per_hour['data_type'] = 'Heart Rate'  # Add data type for legend

# Process stress data if uploaded
if stress_data_zip:
    stress_data_df = process_stress_data_from_zip(stress_data_zip, 'score')
    average_stress_per_day = stress_data_df.groupby('Date').agg(
        Average_stress=('score', 'mean')
    ).reset_index()
    
# Process heart rate data if uploaded
if heart_rate_data_zip:
    heart_rate_data_df = process_heart_rate_data_from_zip(heart_rate_data_zip, 'heart_rate')
    average_hr_per_day = heart_rate_data_df.groupby('Date').agg(
        Average_HR=('heart_rate', 'mean')
    ).reset_index()

if stress_data_zip and heart_rate_data_zip:

    # Merge average stress and heart rate per hour
    combined_hourly_df = pd.concat([average_stress_per_hour, average_heart_rate_per_hour], ignore_index=True)

    # Plot average stress and heart rate per hour
    if 'hour' in combined_hourly_df.columns:
        st.subheader("Average Stress and Heart Rate Per Hour")
        stress_heart_rate_hourly_chart = alt.Chart(combined_hourly_df).mark_circle(size=60).encode(
            x='hour:O',
            y='average_value:Q',
            color=alt.Color('data_type:N', scale=alt.Scale(domain=['Stress', 'Heart Rate'], range=['purple', 'steelblue'])),
            tooltip=['hour:O', 'average_value:Q']
        ).properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(stress_heart_rate_hourly_chart, use_container_width=True)
    else:
        st.warning("No data available to plot.")

    # Process stress data if uploaded
    if stress_data_zip:
        stress_data_df = process_stress_data_from_zip(stress_data_zip, 'score')
        if 'day_of_week' not in stress_data_df.columns:
            st.error("Day of week column missing in stress data.")
        else:
            # Calculate average stress score per day of week
            average_stress_per_day_of_week = stress_data_df.groupby('day_of_week').agg(
                average_value=('score', 'mean')
            ).reset_index()
            average_stress_per_day_of_week['data_type'] = 'Stress'  # Add data type for legend

    # Process heart rate data if uploaded
    if heart_rate_data_zip:
        heart_rate_data_df = process_heart_rate_data_from_zip(heart_rate_data_zip, 'heart_rate')
        if 'day_of_week' not in heart_rate_data_df.columns:
            st.error("Day of week column missing in heart rate data.")
        else:
            # Calculate average heart rate per day of week
            average_heart_rate_per_day_of_week = heart_rate_data_df.groupby('day_of_week').agg(
                average_value=('heart_rate', 'mean')
            ).reset_index()
            average_heart_rate_per_day_of_week['data_type'] = 'Heart Rate'  # Add data type for legend

    # Merge average stress and heart rate per day of week
    combined_daily_of_week_df = pd.concat([average_stress_per_day_of_week, average_heart_rate_per_day_of_week], ignore_index=True)

    # Plot average stress and heart rate per day of week
    if 'day_of_week' in combined_daily_of_week_df.columns:
        st.subheader("Average Stress and Heart Rate Per Day of Week")
        stress_heart_rate_day_of_week_chart = alt.Chart(combined_daily_of_week_df).mark_circle(size=60).encode(
            x=alt.X('day_of_week:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            y='average_value:Q',
            color=alt.Color('data_type:N', scale=alt.Scale(domain=['Stress', 'Heart Rate'], range=['purple', 'steelblue'])),
            tooltip=['day_of_week:N', 'average_value:Q']
        ).properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(stress_heart_rate_day_of_week_chart, use_container_width=True)
    else:
        st.warning("No data available to plot.")

    # Process stress data if uploaded
    if stress_data_zip:
        stress_data_df = process_stress_data_from_zip(stress_data_zip, 'score')
        if 'week_in_year' not in stress_data_df.columns:
            st.error("Week in year column missing in stress data.")
        else:
            # Calculate average stress score per week in year
            average_stress_per_week = stress_data_df.groupby('week_in_year').agg(
                average_value=('score', 'mean'),
                week_start_date=('start_time', lambda x: (x.min() - timedelta(days=x.min().weekday())).strftime('%Y-%m-%d'))
            ).reset_index()
            average_stress_per_week['data_type'] = 'Stress'  # Add data type for legend

    # Process heart rate data if uploaded
    if heart_rate_data_zip:
        heart_rate_data_df = process_heart_rate_data_from_zip(heart_rate_data_zip, 'heart_rate')
        if 'week_in_year' not in heart_rate_data_df.columns:
            st.error("Week in year column missing in heart rate data.")
        else:
            # Calculate average heart rate per week in year
            average_heart_rate_per_week = heart_rate_data_df.groupby('week_in_year').agg(
                average_value=('heart_rate', 'mean'),
                week_start_date=('start_time', lambda x: (x.min() - timedelta(days=x.min().weekday())).strftime('%Y-%m-%d'))
            ).reset_index()
            average_heart_rate_per_week['data_type'] = 'Heart Rate'  # Add data type for legend

    # Merge average stress and heart rate per week in year
    combined_weekly_df = pd.concat([average_stress_per_week, average_heart_rate_per_week], ignore_index=True)

    # Plot average stress and heart rate per week in year
    if 'week_in_year' in combined_weekly_df.columns:
        st.subheader("Average Stress and Heart Rate Per Week in Year")
        stress_heart_rate_weekly_chart = alt.Chart(combined_weekly_df).mark_circle(size=60).encode(
            x=alt.X('week_start_date:T', axis=alt.Axis(title='Week Start Date')),
            y='average_value:Q',
            color=alt.Color('data_type:N', scale=alt.Scale(domain=['Stress', 'Heart Rate'], range=['purple', 'steelblue'])),
            tooltip=['week_start_date:T', 'average_value:Q']
        ).properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(stress_heart_rate_weekly_chart, use_container_width=True)
    else:
        st.warning("No data available to plot.")

# Process data to create dataframe_days
if awt_uploaded_file is not None and google_maps_uploaded_files and survey_uploaded_file and stress_data_zip and heart_rate_data_zip is not None:
    try:
        # Initialize a list to store the day-wise data
        day_rows = []

        # Get unique dates from Begin column of dataframe_merged_awt_with_locations
        unique_dates = pd.to_datetime(dataframe_merged_awt_with_locations['Begin']).dt.date.unique()

        # Iterate over each unique date
        for date in unique_dates:
            # Filter dataframe_merged_awt_with_locations for the current date
            filtered_data = dataframe_merged_awt_with_locations[pd.to_datetime(dataframe_merged_awt_with_locations['Begin']).dt.date == date]

            # Get start time (minimum Begin) and end time (maximum End) for the day
            start_time = filtered_data['Begin'].min()
            end_time = filtered_data['End'].max()

            started_day = start_time.hour + start_time.minute/60.0
            ended_day = end_time.hour + end_time.minute/60.0

            # Calculate total computer time for the day in hours (with two decimal places)
            total_computer_time = (pd.to_datetime(filtered_data['End']) - pd.to_datetime(filtered_data['Begin'])).sum().total_seconds() / 3600

            # Calculate number of computer breaks and total duration of breaks
            breaks_count = 0
            breaks_duration = 0

            # Iterate through rows to find breaks
            for index in range(len(filtered_data) - 1):
                current_end = pd.to_datetime(filtered_data.iloc[index]['End'])
                next_begin = pd.to_datetime(filtered_data.iloc[index + 1]['Begin'])
                if next_begin > current_end:
                    breaks_count += 1
                    breaks_duration += (next_begin - current_end).total_seconds()

            breaks_duration_hours = breaks_duration / 3600

            # Calculate percentage of time spent at home
            home_location = 'Drieboomlaan 273, 1624 BJ Hoorn, Nederland'
            home_time = filtered_data[filtered_data['Location'] == home_location]['End'].sub(filtered_data[filtered_data['Location'] == home_location]['Begin']).sum().total_seconds() / 3600
            percentage_at_home = (home_time / total_computer_time) if total_computer_time > 0 else 0

            # Calculate percentage of time spent at office
            office_location = 'Heidelberglaan 8, 3584 CS Utrecht, Nederland'
            office_time = filtered_data[filtered_data['Location'] == office_location]['End'].sub(filtered_data[filtered_data['Location'] == office_location]['Begin']).sum().total_seconds() / 3600
            percentage_at_office = (office_time / total_computer_time) if total_computer_time > 0 else 0

            # Check if date exists in survey data
            if date.strftime('%Y-%m-%d') in dataframe_survey['Date'].values:
                # Get survey data for the date
                survey_data = dataframe_survey[dataframe_survey['Date'] == date.strftime('%Y-%m-%d')].iloc[0]
                productivity = survey_data['Productivity'] / 5.0 if not pd.isna(survey_data['Productivity']) else None
                vigor = survey_data['Vigor'] / 7.0 if not pd.isna(survey_data['Vigor']) else None
                dedication = survey_data['Dedication'] / 7.0 if not pd.isna(survey_data['Dedication']) else None
                absorption = survey_data['Absorption'] / 7.0 if not pd.isna(survey_data['Absorption']) else None
            else:
                # Set survey data to None if not available
                productivity = None
                vigor = None
                dedication = None
                absorption = None

            # Calculate work engagement as average of vigor, dedication, absorption
            work_engagement = np.mean([vigor, dedication, absorption]) if not (pd.isna(vigor) or pd.isna(dedication) or pd.isna(absorption)) else None

            # Append the day data to the list
            day_rows.append({
                "Date": date,
                "Start_time": start_time,
                "Day_started": started_day,
                "End_time": end_time,
                "Day_ended": ended_day,
                "Total_computer_time": round(total_computer_time, 2),
                "Computer_breaks_num": breaks_count,
                "Computer_breaks_total_duration": round(breaks_duration_hours, 2),
                "Locations": "; ".join(filtered_data['Location'].unique()),
                "Percentage_at_home": round(percentage_at_home, 2),
                "Percentage_at_office": round(percentage_at_office, 2),
                "Productivity": productivity,
                "Vigor": vigor,
                "Dedication": dedication,
                "Absorption": absorption,
                "Work_engagement": work_engagement
            })

        # Create a new DataFrame with the day-wise data
        dataframe_days = pd.DataFrame(day_rows)

        # Merge average stress and heart rate data with dataframe_days
        dataframe_days = pd.merge(dataframe_days, average_stress_per_day, on='Date', how='left')
        dataframe_days = pd.merge(dataframe_days, average_hr_per_day, on='Date', how='left')

        # Display the dataframe_days in Streamlit
        st.write("Final dataframe_days:")
        st.write(dataframe_days)

    except Exception as e:
        st.error(f"An error occurred while creating dataframe_days: {e}")

    # Filter out non-numeric columns
    numeric_columns = dataframe_days.select_dtypes(include=[np.number]).columns

    # Compute correlation matrix for numeric columns only
    correlation_matrix = dataframe_days[numeric_columns].corr()

    # Displaying the correlation matrix using streamlit
    st.write("## Correlation Matrix")
    st.write(correlation_matrix)

    # Function to interpret correlation scores
    def interpret_correlation_score(r, var1, var2):
        if r > 0.7:
            return f"Strong Positive Correlation: As `{var1}` increases, there is a strong tendency for `{var2}` to also increase, but not necessarily at a constant rate."
        elif r > 0.3:
            return f"Moderate Positive Correlation: As `{var1}` increases, there is a noticeable tendency for `{var2}` to also increase."
        elif r < -0.7:
            return f"Strong Negative Correlation: As `{var1}` increases, there is a strong tendency for `{var2}` to decrease at a consistent rate."
        elif r < -0.3:
            return f"Moderate Negative Correlation: As `{var1}` increases, there is a noticeable tendency for `{var2}` to decrease."
        else:
            return "No strong or moderate correlation."

    # Displaying a styled heatmap of the correlation matrix using HTML/CSS
    st.write("## Heatmap of Correlation Matrix")

    # Define a function to generate HTML for heatmap with column labels
    def heatmap_html(data):
        n = len(data)
        html = f'<table style="border-collapse: collapse; border: none; font-size: 12px;">'
        
        # Header row with rotated column labels
        html += '<tr>'
        html += '<td style="border: none;"></td>'  # Empty cell for row labels column
        for col in data.columns:
            html += f'<td style="border: none; padding: 8px; writing-mode: vertical-lr; transform: rotate(180deg);">{col}</td>'
        html += '</tr>'
        
        for i in range(n):
            html += '<tr>'
            html += f'<td style="border: none; padding: 8px; font-weight: bold;">{data.columns[i]}</td>'  # Row label
            for j in range(n):
                if j > i:  # Display only upper triangle of the matrix
                    value = data.iloc[i, j]
                    color = 'background-color: lightblue;' if i == j else f'background-color: rgba(100, 149, 237, {abs(value) / 1.5});'
                    html += f'<td style="border: none; padding: 8px; {color}">{value:.2f}</td>'
                else:
                    html += '<td style="border: none;"></td>'
            html += '</tr>'
        html += '</table>'
        return html

    # Display the heatmap using HTML
    st.write(heatmap_html(correlation_matrix), unsafe_allow_html=True)

    # Display interpretations of correlation coefficients
    st.write("## Interpretation of Correlation Scores")
    for i, col in enumerate(correlation_matrix.columns):
        for j, index in enumerate(correlation_matrix.columns):
            if j > i:  # Process only upper triangle of the matrix
                value = correlation_matrix.iloc[i, j]
                interpretation = interpret_correlation_score(value, col, index)
                if "No strong or moderate correlation." not in interpretation:
                    st.write(f"**Correlation between `{col}` and `{index}`:**")
                    st.write(interpretation)
                    # Generate scatterplot for strong and moderate positive and negative correlations
                    if abs(value) > 0.3:
                        chart_data = dataframe_days[[col, index]]
                        chart = alt.Chart(chart_data).mark_circle().encode(
                            x=col,
                            y=index,
                            tooltip=[col, index]
                        ).properties(
                            width=400,
                            height=300
                        )
                        st.altair_chart(chart)
