import numpy as np
import pandas as pd
import streamlit as st
import json
from io import StringIO
from datetime import datetime, timedelta
import re
import csv

"""
# Let's analyse your AWT data!
"""

# Sidebar for accepting input parameters
with st.sidebar:
    # Load AWT data
    st.header('Load your data')
    st.markdown('**1. AWT data**')
    awt_uploaded_file = st.file_uploader("Upload your Tockler data here. You can export your data by going to Tockler > Search > Set a time period > Export to CSV.")

    # Load Google Maps data
    st.markdown('**2. Google Maps Timeline**')
    google_maps_uploaded_file = st.file_uploader("Upload your Google Maps Timeline data here. You can export your data by going to https://myaccount.google.com/yourdata/maps?hl=en > Download your Maps data > Check Location History (Timeline). There should be one JSON file per month.")

    # Load Survey results data
    st.markdown('**3. Survey results**')
    survey_uploaded_file = st.file_uploader("Upload your survey results here. The CSV should contain 5 columns: Date, Productivity, Vigor, Dedication, Absorption.")

# Main section for processing AWT data
if awt_uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a dataframe
        awt_stringio = StringIO(awt_uploaded_file.getvalue().decode('latin1'))
        dialect = csv.Sniffer().sniff(awt_stringio.read(1024))
        awt_stringio.seek(0)
        dataframe_awt = pd.read_csv(awt_stringio, delimiter=dialect.delimiter)

        # Drop the 'Type' column if it exists
        if 'Type' in dataframe_awt.columns:
            dataframe_awt = dataframe_awt.drop(columns=['Type'])

        # Display the first 5 rows of the dataframe
        st.write("Snippet of the raw AWT data:")
        st.write(dataframe_awt.head())

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

# Check if a Google Maps file has been uploaded 
if google_maps_uploaded_file is not None:
    # Read the JSON file
    maps_data = json.load(google_maps_uploaded_file)

    # Initialize an empty list to store the place visits
    place_visits = []

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
    st.write("Google Maps locations visited:")
    st.write(dataframe_locations)

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
    st.write("Enriched AWT data with locations:")
    st.write(dataframe_merged_awt_with_locations)

# Check if a Survey results file has been uploaded
if survey_uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a dataframe
        survey_stringio = StringIO(survey_uploaded_file.getvalue().decode('utf-8'))
        dialect = csv.Sniffer().sniff(survey_stringio.read(1024))
        survey_stringio.seek(0)
        dataframe_survey = pd.read_csv(survey_stringio, delimiter=dialect.delimiter)

        # Display the first 5 rows of the dataframe
        st.write("Snippet of the survey results data:")
        st.write(dataframe_survey.head())

        # Convert survey date format to match
        dataframe_survey['Date'] = pd.to_datetime(dataframe_survey['Date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

    except pd.errors.ParserError as e:
        st.error(f"Error parsing Survey CSV file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Process data to create dataframe_days
if awt_uploaded_file is not None and google_maps_uploaded_file is not None and survey_uploaded_file is not None:
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
                "End_time": end_time,
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

        # Create the dataframe_days DataFrame
        dataframe_days = pd.DataFrame(day_rows)

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
        #elif r > 0.3:
            return f"Moderate Positive Correlation: As `{var1}` increases, there is a noticeable tendency for `{var2}` to also increase."
        elif r < -0.7:
            return f"Strong Negative Correlation: As `{var1}` increases, there is a strong tendency for `{var2}` to decrease at a consistent rate."
        #elif r < -0.3:
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
                value = data.iloc[i, j]
                color = 'background-color: lightblue;' if i == j else f'background-color: rgba(100, 149, 237, {abs(value) / 1.5});'
                html += f'<td style="border: none; padding: 8px; {color}">{value:.2f}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    # Display the heatmap using HTML
    st.write(heatmap_html(correlation_matrix), unsafe_allow_html=True)

    # Display interpretations of correlation coefficients
    st.write("## Interpretation of Correlation Scores")
    for col in correlation_matrix.columns:
        for index, value in correlation_matrix[col].items():
            if col != index:
                interpretation = interpret_correlation_score(value, col, index)
                if "No strong or moderate correlation." not in interpretation:
                    #st.write(f"**Correlation between `{col}` and `{index}`:**")
                    st.write(interpretation)