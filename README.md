# Predicting the results of football matches
### The essence of the project is to develop a model that can predict the outcome of a match for two teams, taking into account the following distinctive features: 
- **teams play only within their own league, but the model should be able to predict including the result for two teams from different leagues**
- **the matches are divided by time, respectively, the model should give an up-to-date result for teams that are within the same season**
- **for some teams, the data is incomplete, that is, there are not all statistics for 38 games for the season**
____
The file with the source database could not be uploaded to the repository, but it can be obtained by downloading it from the kaggle website (url: https://www.kaggle.com/code/jacobopedrosa/simple-football-predictions-accurance-75-40/data)
____
In a text file `sql_code.txt` contains queries for a combination of tables from the source database. Accordingly , in the file `match_info.csv` information on the results of each match is generated. The file `team_skills.csv` displays the attributes of the commands, `player_skills.scv` - displays the attributes of the players.

In the file `Football_matches_results_prediction.ipynb` the code for solving the problem is given. The stages of work, possible models, results and comments can be found there
 
