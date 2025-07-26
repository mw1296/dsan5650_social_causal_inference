import pandas as pd
import os
os.getcwd()
df_dic = {}

def create_df_race(folder_path, file_name, demographic_type, rows, id_vars, var_name, value_name):
    file_path = os.path.join("final_project/data", folder_path, file_name + ".xlsx")    
    df = pd.read_excel(file_path, skiprows=rows)
    df_long = df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
    df_long[demographic_type] = file_name
    df_dic[file_name] = df_long
    return df_dic


folder_path = ["unemployment_rate_by_race",
             "unemployment_rate_by_sex",
             "unemployment_rate_by_industry",
             "unemployment_rate_by_age",
             "unemployment_rate_by_education_attainment"]
demographic_type = ['Race', 'Sex', 'Industry', 'Age', 'Education_attainment']

for i in range(0, len(folder_path)):
    print(folder_path[i])
    file_name = [f.removesuffix('.xlsx') for f in os.listdir(os.path.join("final_project/data", folder_path[i])) if f.endswith('.xlsx')]

demographic_type = 'Race'
d_vars="Year"
var_name="Month"
value_name="Unemployment_rate"
rows = 12
for f in file_name:
    create_df_race(folder_path, f, demographic_type, rows, "Year", "Month", "Unemployment_rate")


#file_path = os.path.join("final_project/data", folder_path, file_name + ".xlsx")
# rows = 12
# df = pd.read_excel(file_path, skiprows=rows)
# df_long = df.melt(id_vars="Year", var_name="Month", value_name="Unemployment_rate")
# df_long["Race"] = "white"
# df_dic[file_name] = df_long


