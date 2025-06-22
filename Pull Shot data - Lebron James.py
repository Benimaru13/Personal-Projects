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
lbj_shot_data = pd.read_csv("lebron_shots_2024_25.csv")
#print(lbj_shot_data.tail(10))
print(lbj_shot_data.shape)
