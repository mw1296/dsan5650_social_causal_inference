import pandas as pd
import os
os.getcwd()
#os.chdir('../pgm-unemployment-rate-and-durations')

def create_df_race(folder_path, demographic_type, id_vars, var_name, value_name):
    merged_df = pd.DataFrame()
    file_name = [f.removesuffix('.xlsx') for f in os.listdir(os.path.join("../data", folder_path)) if f.endswith('.xlsx')]
    for f in file_name:
        file_path = os.path.join("../data", folder_path, f + ".xlsx")    
        df = pd.read_excel(file_path)
        rows = df[df.iloc[:,0]=='Year'].index[0]+2-1
        df = pd.read_excel(file_path, skiprows=rows)
        df_long = df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
        df_long[demographic_type] = f
        merged_df = pd.concat([merged_df, df_long], ignore_index=True)
    return merged_df

folder_path_list = ["unemployment_rate_by_race",
             "unemployment_rate_by_sex",
             "unemployment_rate_by_industry",
             "unemployment_rate_by_age",
            "unemployment_rate_by_education_attainment"]

demographic_type_list = ['Race',
                    'Sex',
                    'Industry',
                    'Age',
                    'Education_attainment']

d_vars="Year"
var_name="Month"
value_name="Unemployment_rate"

df_dic = {}
for i in range(0, len(folder_path_list)):
    folder_path = folder_path_list[i]
    demographic_type=demographic_type_list[i]
    merged_df = create_df_race(folder_path, demographic_type, "Year", "Month", "Unemployment_rate")
    df_dic[demographic_type] = merged_df
display(df_dic['Race'].head(10).style.hide(axis='index').format({'Unemployment_rate':'{:.2f}'}))

import matplotlib.pyplot as plt
import numpy as np

def plot_hist(df_plot, demographic_type):
    unique_categories = df_plot[demographic_type].unique()
    num_unique_categories = len(unique_categories)
    ncol = 3
    nrow = np.int64(np.ceil(num_unique_categories/ncol))
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=True, figsize = (16,nrow*5))
    axes = axes.flatten()
    min_rate = df_plot['Unemployment_rate'].min()
    max_rate = df_plot['Unemployment_rate'].max()
    num_bins = 20 
    bins = np.linspace(min_rate, max_rate, num_bins + 1)
    for i, category in enumerate(unique_categories):
        ax = axes[i]
        vals = df_plot[df_plot[demographic_type]==category]['Unemployment_rate']
        ax.hist(vals, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(vals), color = 'black', linestyle = "--")
        ax.set_title(category, fontsize=12)
        ax.set_xlabel('Unemployment Rate', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
    plt.tight_layout()
    plt.suptitle(f'Unemployment Rate Distribution by {demographic_type}', y=1.02, fontsize=18) 
    plt.show()
    
# Show the graphs
for i in range(0, len(df_dic.keys())):
    df_plot = list(df_dic.values())[i]
    demographic_type = list(df_dic.keys())[i]
    print(f"Unemployment Rate Distribution Histograms By {demographic_type}")
    plot_hist(df_plot, demographic_type)


# Data pre-processing
# 2. Load and process monthly unemployment of all population
import pandas as pd
import numpy as np
df_overall = pd.read_excel("../data/unemployment_rate_monthly_data.xlsx", skiprows=11)
df_overall_long = df_overall.melt(id_vars="Year", var_name="Month", value_name="Unemployment_rate")


# 3. process the survey data
df_survey_jun = pd.read_csv("../data/survey_data/jun_2025.csv")
df_survey_jun.columns= ["sex","education_attainment","race","age","employment_status",
                        "unmployment_duration","industry","occupation","industry_detailed",
                        "occupation_detailed"]
## create mapping tables to convert the demographic codes to the descriptions
sex_mapping = {
        1: "male",
        2: "female"
}

education_attainment_codes = list(np.arange(31,47))
education_attainment_values = [
        'less_than_high_school',
		'less_than_high_school',
		'less_than_high_school',
        'less_than_high_school',
'less_than_high_school',
'less_than_high_school',
'less_than_high_school',
'less_than_high_school',
'high_school',
'some_college_or_associate_degree',
'some_college_or_associate_degree',
'some_college_or_associate_degree',
'bachelors_degree',
'masters_degree',
'professional_degree',
'doctoral_degree'
    ]    
education_attainment_mapping=dict(zip(education_attainment_codes,education_attainment_values))

race_codes = list(np.arange(1,27))
race_values = [
        "white",
        "black",
        "other",
        "asian"
    ] + ["other"] * 22
race_mapping= dict(zip(race_codes, race_values))

employment_status_codes =[
        1,2,3,4,5,6,7
    ]
employment_status_names = [
        "employed",
        "employed",
        "unemployed",
        "unemployed",
        "not in labor force",
        "not in labor force",
        "not in labor force"
    ]
employment_status_mapping = dict(zip(employment_status_codes,employment_status_names))

industry_codes = list(np.arange(1,15))
industry_names = [
    "Agriculture, forestry, fishing, and hunting",
    "Mining",
    "Construction",
    "Manufacturing",
    "Wholesale and retail trade",
    "Transportation and utilities",
    "Information",
    "Financial activities",
    "Professional and business services",
    "Educational and health services",
    "Leisure and hospitality",
    "Other services",
    "Public administration",
    "Armed Forces"
    ]
industry_mapping = dict(zip(industry_codes, industry_names))
## Create new columns with decoded names
df_survey_jun['sex_name'] = df_survey_jun['sex'].map(sex_mapping)
df_survey_jun['race_name'] = df_survey_jun['race'].map(race_mapping)
df_survey_jun['education_attainment_name'] = df_survey_jun['education_attainment'].map(education_attainment_mapping)
df_survey_jun['employment_status_name'] = df_survey_jun['employment_status'].map(employment_status_mapping)
df_survey_jun['industry_name'] = df_survey_jun['industry'].map(industry_mapping)
## Filter out industry_name na rows
df_survey_jun = df_survey_jun[~df_survey_jun['industry_name'].isna()]
## Add column is_black_african for the binary model
df_survey_jun['is_black_african'] = (df_survey_jun['race_name']=='black').astype(int)
## Add column is_asian for the binary model
df_survey_jun['is_asian'] = (df_survey_jun['race_name']=='asian').astype(int)
## Add column is_asian for the binary model
df_survey_jun['is_white'] = (df_survey_jun['race_name']=='white').astype(int)
## Add column employment_status for the binary model
df_survey_jun['unemployment_status'] = (df_survey_jun['employment_status_name']=='unemployed').astype(int)

## Show the table with selective columns
display(df_survey_jun.loc[:,[col for col in df_survey_jun.columns if 'name' in col or col == 'is_black_african'  or col == 'is_white'  or col == 'is_asian']].head(10).style.hide(axis='index'))

## Process monthly survey data of 2024 Jan-Jul and 2025 Jan-Jul
survey_file_names = [f.removesuffix(".csv") for f in os.listdir('../data/survey_data') if f.endswith("csv")]
cols = ['prmjind1', 'pemlr'] # keep columns industry and employment_status
survey_df = []
for s in survey_file_names:
    survey_file = pd.read_csv(os.path.join('../data/survey_data/', s + ".csv" ))
    survey_data = survey_file[cols]
    survey_data.columns=['industry','employment_status']
    survey_data['month_year'] = s
    survey_df.append(survey_data)
# Combine all into a single DataFrame
survey_df = pd.concat(survey_df, ignore_index=True)
survey_df['month_year'].unique()
# decode the industry and employment_status columns
survey_df['industry_name'] = survey_df['industry'].map(industry_mapping)
survey_df = survey_df[~survey_df['industry_name'].isna()]
survey_df['employment_status_description'] = survey_data['employment_status'].map(employment_status_mapping)
display(survey_df.head(10).style.hide(axis='index'))


# Fit the log-norma distribution# Fit log-norm distribution using population unemployment rate data
from scipy.stats import lognorm, kstest, skew
## Get the unemployment_rate data that is not NA, convert to array
data_to_fit = df_overall_long[~np.isnan(np.array(df_overall_long['Unemployment_rate']))]['Unemployment_rate']
## Get the estimated parameters
sigma_fit, loc_fit, scale_fit = lognorm.fit(data_to_fit)
print(f"\nFitted Log-Norm Distribution Parameters:")
print(f"  Shape parameter 'alpha'): {sigma_fit:.2f}")
print(f"  Location parameter (loc): {loc_fit:.2f}")
print(f"  Scale parameter (scale): {scale_fit:.2f}")
# Visualize the fit
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
## Histogram of the unemployment data
ax.hist(data_to_fit, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data Histogram')
## Fitted histogram
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = lognorm.pdf(x, sigma_fit, loc=loc_fit, scale=scale_fit)
ax.plot(x, p, 'darkred', lw=2, label='Fitted Log-Normal PDF')
ax.set_xlabel('Unemployment Rate', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.suptitle("Data Histogram vs. Fitted Log-Normal Distribution PDF", y=1.02, fontsize=18)
plt.show()


# Fit gamma distribution using MLE
from scipy.stats import gamma, skew, kstest
alpha_fit, loc_fit, scale_fit = gamma.fit(data_to_fit) # fit gamma distribution
# Print the estimated parameters
print(f"\nFitted Gamma Distribution Parameters:")
print(f"  Shape parameter 'alpha': {alpha_fit:.2f}")
print(f"  Location parameter 'loc': {loc_fit:.2f}")
print(f"  Scale parameter 'scale'): {scale_fit:.2f}")
# Visualize the fit
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
## Histogram of the unemployment data
ax.hist(data_to_fit, bins = 20, density=True, color='skyblue', alpha= 0.7, edgecolor='black', label = 'Data Histogram')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
## Plot fitted gamma pdf
p = gamma.pdf(x, alpha_fit, loc=loc_fit, scale=scale_fit)
ax.plot(x, p, 'darkred', lw=2, label='Fitted Gamma PDF')
ax.set_xlabel('Unemployment Rate', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.suptitle("Data Histogram vs. Fitted Gamma Distribution PDF", y=1.02, fontsize=18)
plt.show()
# Check the p value of fitted gamma distribution
d_statistic, p_value = kstest(data_to_fit, lambda x: gamma.cdf(x, alpha_fit, loc=loc_fit, scale=scale_fit))
print(f"\nKolmogorov-Smirnov Test Results:")
print(f"  D-statistic: {d_statistic:.4f}")
print(f"  P-value: {p_value:.4f}")

# Fit Weibull distribution using MLE
from scipy.stats import weibull_min, skew, kstest
from scipy.special import gamma as gamma_func # For Weibull mean calculation
c_fit, loc_fit, scale_fit = weibull_min.fit(data_to_fit) # fit weibull distribution
# Print the estimated parameters
print(f"\nFitted Weibull Distribution Parameters:")
print(f"  Shape parameter 'c' (k): {c_fit:.4f}")
print(f"  Location parameter (loc): {loc_fit:.4f}")
print(f"  Scale parameter (scale, lambda): {scale_fit:.4f}")
# Visualize the fit
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
## Histogram of the unemployment data
ax.hist(data_to_fit, bins = 20, density=True, color='skyblue', alpha= 0.7, edgecolor='black', label = 'Data Histogram')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
## Plot fitted gamma pdf
p = weibull_min.pdf(x, c_fit, loc=loc_fit, scale=scale_fit)
ax.plot(x, p, 'darkred', lw=2, label='Fitted Weibull PDF')
ax.set_xlabel('Unemployment Rate', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.suptitle("Data Histogram vs. Fitted Weibull Distribution PDF", y=1.02, fontsize=18)
plt.show()
# Fit the Weibull
d_statistic, p_value = kstest(data_to_fit, lambda x: weibull_min.cdf(x, c_fit, loc=loc_fit, scale=scale_fit))
print(f"\nKolmogorov-Smirnov Test Results:")
print(f"  D-statistic: {d_statistic:.4f}")
print(f"  P-value: {p_value:.4f}")
# Calculate inferred mean
fitted_mean = loc_fit + scale_fit * gamma_func(1 + 1/c_fit)

#| tbl-cap: "Weibull Distribution and Fitness Statistics"
#| tbl-col: true
#| tbl-center: true
# Print out the Weibull stats
# Print the weibull stats
weibull_stats = {
  "parameters": [
    "Shape (c)",
    "Location (loc)" ,
    "Scale",
    "D_statistics",
    "P-value",
    "Fitted Mean"
  ],
  "values": [
     round(c_fit, 4),
     round(loc_fit, 4),
     round(scale_fit, 4),
     round(d_statistic, 4),
     round(p_value, 4),
     round(fitted_mean, 2)
  ]
}
weibull_stats_table = pd.DataFrame(weibull_stats)
display(weibull_stats_table.style.hide(axis='index'))


### population unemployment mean probability from the fitted Weibull distribution
weibull_fitted_mean_prob = 0.0435
# Convert the fitted mean probability to its log-odds equivalent in order to model binary outcome with logistic regression
population_mean_log_odds = np.log(weibull_fitted_mean_prob / (1 - weibull_fitted_mean_prob))

# Find the means of different demographics
def mean_prob_demographic(df_name, demographic_type, demographic_description):
    demographic_mean_prob = df_dic[df_name].loc[df_dic[df_name][demographic_type]==demographic_description]['Unemployment_rate'].mean()/100
    demographic_mean_log_odds = np.log(demographic_mean_prob / (1 - demographic_mean_prob))
    # beta_effect is the difference between the demographic group mean log-odds and the population mean log-odds
    beta_effect = demographic_mean_log_odds - population_mean_log_odds
    return beta_effect

# Create a grid of demographic information to return the means
demographic_grid = pd.DataFrame({
    "df_name":[
        'Race',
        'Race',
        'Race'
    ],
    "demographic_type":[
        "Race", 
        "Race",
        "Race"
    ],
    "demographic_description":[
        'black_and_african',
        'white',
        'asian'
    ]
})
# Return the means
demographic_grid['beta_effect']= np.float64(0)
for row in range(0,len(demographic_grid)):
    # Mean of demographic groups
    beta_effect = mean_prob_demographic(demographic_grid['df_name'][row], demographic_grid['demographic_type'][row], demographic_grid['demographic_description'][row])
    demographic_grid['beta_effect'][row] = beta_effect

# Create the pymc model
import pymc as pm
import arviz as az

with pm.Model() as unemployment_race_model:
    # Prior of population mean log odds
    mu_population_log_odds = pm.Normal('mu_population_log_odds', mu=population_mean_log_odds, sigma=0.5)
    
    # Additive race effect parameter on population mean
    beta_black = pm.Normal('beta_black', mu=demographic_grid[demographic_grid['demographic_description']=='black_and_african']['beta_effect'], sigma=0.3) # Can be positive or negative

    beta_asian = pm.Normal('beta_asian', mu=demographic_grid[demographic_grid['demographic_description']=='asian']['beta_effect'], sigma=0.3) 

    beta_white = pm.Normal('beta_white', mu=demographic_grid[demographic_grid['demographic_description']=='white']['beta_effect'], sigma=0.3) 
    
    # noise
    sigma = pm.HalfNormal('sigma', sigma=0.2)

    # log-odds and probabilities for each racial group
    mu_white_log_odds = pm.Deterministic('mu_white_log_odds', mu_population_log_odds + beta_white)
    mu_black_log_odds = pm.Deterministic('mu_black_log_odds', mu_population_log_odds + beta_black)
    mu_asian_log_odds = pm.Deterministic('mu_asian_log_odds', mu_population_log_odds + beta_asian)

    p_population = pm.Deterministic('p_population', pm.math.invlogit(mu_population_log_odds))
    p_white = pm.Deterministic('p_white', pm.math.invlogit(mu_white_log_odds))
    p_black = pm.Deterministic('p_black', pm.math.invlogit(mu_black_log_odds))
    p_asian = pm.Deterministic('p_asian', pm.math.invlogit(mu_asian_log_odds))

    # Likelihoods for observed data: use parameters like derived above to make predictions about the "true" probabilities of unemployment.

    # predict population mean
    population_rates_df = df_overall_long[~np.isnan(df_overall_long['Unemployment_rate'])]
    population_rates_df['rate_logit'] = np.log(population_rates_df['Unemployment_rate']*0.01/(1-population_rates_df['Unemployment_rate']*0.01))
    population_rates_obs = pm.Normal(
        'population_rates_obs',
        mu=mu_population_log_odds, 
        sigma=sigma,
        observed=population_rates_df['rate_logit'].values
    )

    # predict black mean
    black_rates_df = df_dic['Race'][df_dic['Race']['Race']=='black_and_african']
    black_rates_df = black_rates_df[~np.isnan(black_rates_df['Unemployment_rate'])] 
    black_rates_df['rate_logit'] = np.log(black_rates_df['Unemployment_rate']*0.01/(1-black_rates_df['Unemployment_rate']*0.01))
    black_rates_obs = pm.Normal(
        'black_rates_obs',
        mu=mu_black_log_odds, 
        sigma=sigma,
        observed=black_rates_df['rate_logit'].values
    )

    # predict asian mean
    asian_rates_df = df_dic['Race'][df_dic['Race']['Race']=='asian']
    asian_rates_df = asian_rates_df[~np.isnan(asian_rates_df['Unemployment_rate'])] 
    asian_rates_df['rate_logit'] = np.log(asian_rates_df['Unemployment_rate']*0.01/(1-asian_rates_df['Unemployment_rate']*0.01))
    asian_rates_obs = pm.Normal(
        'asian_rates_obs',
        mu=mu_asian_log_odds,
        sigma=sigma,
        observed=asian_rates_df['rate_logit'].values
    )

    # predict white mean
    white_rates_df = df_dic['Race'][df_dic['Race']['Race']=='white']
    white_rates_df = white_rates_df[~np.isnan(white_rates_df['Unemployment_rate'])] 
    white_rates_df['rate_logit'] = np.log(white_rates_df['Unemployment_rate']*0.01/(1-white_rates_df['Unemployment_rate']*0.01))
    white_rates_obs = pm.Normal(
        'white_rates_obs',
        mu=mu_white_log_odds,
        sigma=sigma,
        observed=white_rates_df['rate_logit'].values
    )
    
    # Assign the correct probability based on race
    p_individual_for_obs = pm.math.switch(
        df_survey_jun['is_black_african'].values,
        p_black, # if is_black, use black prob
        pm.math.switch(
            df_survey_jun['is_asian'].values, 
            p_asian, # if is_asian, use asian prob
            p_white # else, use white prob
        )
    )
    individual_status_obs = pm.Bernoulli(
        'individual_status_obs',
        p=p_individual_for_obs,
        observed=df_survey_jun['unemployment_status'].values
    )
    
pm.model_to_graphviz(unemployment_race_model)

# Fit the Model (Perform MCMC Sampling)
print("--- Starting MCMC sampling... ---")
with unemployment_race_model:
    indiv_trace = pm.sample(random_seed=5650)

# Analyze Results and Visualize
print("\n--- Model Summary (Simplified Model - Multiple Races) ---")
display(pd.DataFrame(az.summary(indiv_trace, var_names=['mu_population_log_odds', 'beta_black', 'beta_asian', 'beta_white', 'sigma'])))
fig_trace = az.plot_trace(indiv_trace)
plt.suptitle('Trace plot', fontsize=16)
plt.subplots_adjust(hspace=0.5) 

# Convert to DataFrame
indiv_post_df = indiv_trace.posterior.to_dataframe().reset_index()
# Extract only the first chain
indiv_post_df = indiv_post_df[indiv_post_df['chain'] == 0]
# Melt as before
indiv_post_df = indiv_post_df[['draw','beta_black', 'beta_asian', 'beta_white']].melt(id_vars=['draw'])
# And plot!
import seaborn as sns
import patchworklib as pw
ax = pw.Brick(figsize=(4, 2.5));
sns.kdeplot(
    x="value", hue="variable", data=indiv_post_df, ax=ax,
    fill=True, bw_adjust=2,
);
ax.set_xlabel("Log-Odds Deviation from Overall Mean")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Race Effect Deviations")
ax.savefig()
