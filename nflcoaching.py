# %%
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

nfl_pbp = nfl.import_pbp_data([2018, 2019, 2020, 2021, 2022, 2023])


pbp_rp = nfl_pbp[(nfl_pbp['pass'] == 1) | (nfl_pbp['rush'] == 1) | (nfl_pbp['special_teams_play'] == 1)]
pbp_rp = pbp_rp.dropna(subset=['epa', 'posteam', 'defteam', 'season'])



pass_epa = pbp_rp[(pbp_rp['pass'] == 1)].groupby(['posteam', 'season'])['epa'].mean().reset_index().rename(columns = {'epa' : 'pass_epa'})
rush_epa = pbp_rp[(pbp_rp['rush'] == 1)].groupby(['posteam', 'season', ])['epa'].mean().reset_index().rename(columns = {'epa' : 'rush_epa'})
st_epa = pbp_rp[(pbp_rp['special_teams_play'] == 1)].groupby(['posteam', 'season'])['epa'].mean().reset_index().rename(columns = {'epa' : 'st_epa'})
epa = pd.merge(pass_epa, rush_epa, on=['posteam', 'season'], suffixes=('_pass', '_rush'))
epa = pd.merge(st_epa, epa, on=['posteam', 'season'])



def_epa = pbp_rp[(pbp_rp['pass'] == 1) |(pbp_rp['rush'] == 1)].groupby(['defteam', 'season'])['epa'].mean().reset_index().rename(columns = {'epa' : 'def_epa'})
def_epa.rename(columns={'defteam': 'posteam'}, inplace=True)

epa = pd.merge(epa, def_epa, on = ['posteam', 'season'])

season_data = nfl.import_schedules([2018, 2019, 2020, 2021, 2022, 2023])



season_data['home_win'] = np.where(season_data['home_score'] > season_data['away_score'], 1, 0)
season_data['away_win'] = np.where(season_data['away_score'] > season_data['home_score'], 1, 0)


# %%

teams = pd.concat([season_data['home_team'], season_data['away_team']]).unique()


win_stats = pd.DataFrame({'team': teams})

def calculate_stats(team):
    team_games = season_data[(season_data['home_team'] == team) | (season_data['away_team'] == team)]
    
  
    games_played = team_games.groupby('season').size()
    total_wins = team_games[(team_games['home_team'] == team) & (team_games['home_score'] > team_games['away_score']) |
                            (team_games['away_team'] == team) & (team_games['away_score'] > team_games['home_score'])].groupby('season').size()
    
    return pd.DataFrame({
        'season': games_played.index,
        'team': team,
        'games_played': games_played.values,
        'total_wins': total_wins.values
    })


win_stats = pd.concat([calculate_stats(team) for team in teams])


win_stats['win_percentage'] = (win_stats['total_wins'] / win_stats['games_played']) * 100


print(win_stats)


epa.rename(columns={'posteam': 'team'}, inplace=True)


stats = pd.merge(epa, win_stats, on=['team', 'season'])


wpa = pbp_rp.groupby(['posteam', 'season'])['wpa'].sum().reset_index().rename(columns = {'wpa' : 'wpa'})

wpa.rename(columns={'posteam': 'team'}, inplace=True)

stats = pd.merge(stats, wpa, on=['team', 'season'])


path = '/Users/markstolte/downloads/nfl_coaching_exp.xlsx'
luck = pd.read_excel(path, sheet_name= "Sheet3")

stats = pd.merge(stats, luck, on=['team', 'season'])


stats.rename(columns = {'Total': 'luck'}, inplace=True)

df = stats


df = df.drop(columns=['team', 'season', 'total_wins', 'games_played'])
df = df.dropna()

X = df[['st_epa', 'pass_epa', 'rush_epa', 'def_epa', 'wpa']]
y = df['win_percentage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate the model
coefficients = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])
intercept = reg.intercept_

X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
summary = model.summary()

print(summary)

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


path = '/Users/markstolte/downloads/nfl_coaching_exp.xlsx'
df_coaching = pd.read_excel(path)


X_predict = df_coaching[['st_epa', 'pass_epa', 'rush_epa', 'def_epa', 'luck']]

predicted_win_percentage = reg.predict(X_predict)


df_coaching['predicted_win_percentage'] = predicted_win_percentage


print("Predicted Win Percentages:\n", df_coaching[['team', 'predicted_win_percentage']])



def get_team_names_and_colors():
    teams = nfl.import_team_desc()

    teams = teams[['team_abbr', 'team_name', 'team_color']]

    return teams


path = '/Users/markstolte/downloads/nfl_coaching_exp.xlsx'
df = pd.read_excel(path, sheet_name='Sheet2')

teams_df = get_team_names_and_colors()


df_merged = pd.merge(df, teams_df, on='team_abbr')

df_merged['Performance'] = pd.to_numeric(df_merged['Performance'], errors='coerce')


plt.figure(figsize=(12, 10))  # Adjust figure size as needed for layout


bars = plt.barh(df_merged['team_name'], df_merged['Performance'], color=df_merged['team_color'])


plt.title('Coaching Performance')
plt.xlabel('Win % above expectation')
plt.ylabel('Team')
plt.xlim(-0.3, df_merged['Performance'].max() * 1.1)  # Adjust xlim for better spacing
plt.ylim(-0.5, len(df_merged) - 0.5)  # Set ylim to align with the number of teams
plt.tight_layout()
plt.show()



nfl_pbp




