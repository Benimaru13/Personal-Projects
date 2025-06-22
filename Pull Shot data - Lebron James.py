from nba_api.stats.endpoints import shotchartdetail
import pandas as pd

lebron_id = 2544  # LeBron James
shot_data = shotchartdetail.ShotChartDetail(
    team_id=0, player_id=lebron_id,
    season_type_all_star='Regular Season',
    season_nullable='2024-25'
).get_data_frames()[0]

shot_data.to_csv("lebron_shots_raw_2024_25.csv", index=False)

# The code retrieves LeBron James' shot chart data for the 2024-25 NBA season
# and saves it to a CSV file named "lebron_shots_2024_25
lbj_shot_data = pd.read_csv("lebron_shots_raw_2024_25.csv")
#print(lbj_shot_data.tail(10))

def clean_shot_data(df):
    
    columns_to_keep = [
        "GAME_ID", "GAME_DATE", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "SHOT_DISTANCE", "LOC_X", "LOC_Y",
        "SHOT_ZONE_BASIC", "ACTION_TYPE", "SHOT_TYPE",
        "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG"]
    # Keep only the relevant columns
    df = df[columns_to_keep]
    
    # Convert 'GAME_DATE' to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
    # Convert 'SHOT_CLOCK' to integer
    #df['SHOT_CLOCK'] = df['SHOT_CLOCK'].astype(int)
    
    # Drop NA values in 'SHOT_MADE_FLAG' and 'SHOT_ATTEMPTED_FLAG'
    #df.dropna(subset=["SHOT_CLOCK"], inplace=True)
    df.dropna(subset=["SHOT_MADE_FLAG"], inplace=True)
    df.dropna(subset=["SHOT_ATTEMPTED_FLAG"], inplace=True)
    
    # Convert 'LOC_X' and 'LOC_Y' to float
    df['LOC_X'] = pd.to_numeric(df['LOC_X'], errors='coerce')
    df['LOC_Y'] = pd.to_numeric(df['LOC_Y'], errors='coerce')    
    # Convert 'SHOT_DISTANCE' to float
    df['SHOT_DISTANCE'] = pd.to_numeric(df['SHOT_DISTANCE'], errors='coerce')    
    # Convert Shot zone and action type to categorical
    df['SHOT_ZONE_BASIC'] = df['SHOT_ZONE_BASIC'].astype('category')
    
    return df

lbj_shot_data = clean_shot_data(lbj_shot_data)
# Save the cleaned data to a new CSV file
lbj_shot_data.to_csv("lebron_shots_cleaned_2024_25.csv", index=False)
# Display a sample of the cleaned data
print(lbj_shot_data.sample(10))