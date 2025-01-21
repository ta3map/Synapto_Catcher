import pandas as pd
import numpy as np
import seaborn as sns
from scikit_posthocs import posthoc_dunn
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from itertools import combinations

# Function to convert p-values ​​to stars
def rankstars(p):
    if not np.isnan(p):
        if p > 0.05:
            return 'ns'
        elif p <= 0.0001:
            return '****'
        elif p <= 0.001:
            return '***'
        elif p <= 0.01:
            return '**'
        elif p <= 0.05:
            return '*'
    else:
        return 'ns'
        
def plot_pairwise_pvalues(local_param_data, groups, local_p_values, groups_x_pos):
    height = local_param_data.quantile(0.95)
    h_step = (local_param_data.quantile(0.95) * 2 - height) / (len(groups) + 1)

    # Filter groups to include only those that exist in local_p_values index and columns
    valid_groups = [group for group in groups if group in local_p_values.index and group in local_p_values.columns]

    for j, group1 in enumerate(valid_groups):
        for k, group2 in enumerate(valid_groups):
            if j < k:
                p_value = local_p_values.loc[group1, group2]
                ranktext = rankstars(p_value)
                if not np.isnan(p_value) and ranktext != 'ns':
                    x1, x2 = groups_x_pos[j], groups_x_pos[k]
                    y, h, col = local_param_data.max() + 1, 1, (0.5, 0.5, 0.5, 0.5)
                    height = height + h_step
                    plt.plot([x1, x2], [height, height], lw=1.5, c=col)
                    plt.plot([x2, x2], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.plot([x1, x1], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.text((x1 + x2) * 0.5, height, ranktext, ha='center', va='bottom', color='k')

                        
def analyze_and_plot_many_graphs(file_path, output_folder, groups, locations, numerical_parameters):
    # Reading data from an Excel file
    data = pd.read_excel(file_path)

    # Filtering data by age groups and locations
    filtered_data = data[(data['Group'].isin(groups)) & (data['selected_location'].isin(locations))]

    # Function to calculate Kruskal-Wallis p-values ​​for each location
    def calculate_kruskal_pvalues(data, parameter, groups, group_col='Group', location_col='selected_location'):
        results = {}
        for location in data[location_col].unique():
            group_data = [data[(data[group_col] == group) & (data[location_col] == location)][parameter].dropna() for group in groups]
            if all(len(g) > 0 for g in group_data):
                stat, p_value = kruskal(*group_data)
                results[location] = p_value
            else:
                results[location] = np.nan
        return results

    # Function for performing pairwise comparisons using Dunn's test
    def calculate_dunn_pvalues(data, parameter, groups, group_col='Group', location_col='selected_location'):
        pairwise_results = {}
        for location in data[location_col].unique():
            location_data = data[data[location_col] == location]
            dunn_results = posthoc_dunn(location_data, val_col=parameter, group_col=group_col, p_adjust='bonferroni')
            pairwise_results[location] = dunn_results
        return pairwise_results


    # Function to plot violins and add p-values
    def plot_violin_with_pvalues(data, parameter, category, hue, dunn_pvalues):
        locations = data[hue].unique()
        palette = sns.color_palette("deep", len(locations))
        groups = data[category].unique()
        groups.sort()
        for i, location in enumerate(locations):
            local_p_values = dunn_pvalues[location]
            
            local_data = data[data[hue] == location]
            
            valid_groups = [group for group in groups if group in local_p_values.index and group in local_p_values.columns]
            
            group_indexes = {valid_groups: index for index, valid_groups in enumerate(valid_groups)}
            
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=category, y=parameter, data=local_data, color=palette[i], fill=False)
            
            scatter_x = local_data[category].map(group_indexes)
            scatter_x = scatter_x + np.random.uniform(-0.2, 0.2, size=len(scatter_x))
            sns.scatterplot(x=scatter_x, y=parameter, data=local_data, color=palette[i])
            
            plt.title(location)
            plt.xlabel(category.replace('_', ' '))
            plt.ylabel(parameter)
            
            local_param_data = local_data[parameter]
            groups_x_pos = list(range(np.size(valid_groups)))
            plot_pairwise_pvalues(local_param_data, valid_groups, local_p_values, groups_x_pos)      
            
            # Remove the frame around the graph
            sns.despine()
            plt.show()

    # Calculate Kruskal-Wallis p-values ​​for each parameter and each location
    kruskal_p_values = {}
    for parameter in numerical_parameters:
        kruskal_p_values[parameter] = calculate_kruskal_pvalues(filtered_data, parameter, groups)

    # Calculate Dunn's test p-values ​​for each parameter and each location
    dunn_p_values = {}
    for parameter in numerical_parameters:
        dunn_p_values[parameter] = calculate_dunn_pvalues(filtered_data, parameter, groups)

    # Plotting graphs for numerical parameters with p-values ​​of the Kruskal-Wallis test and Dunn test
    for parameter in numerical_parameters:
        dunn_pvalues = dunn_p_values[parameter]
        plot_violin_with_pvalues(filtered_data, parameter, 'Group', 'selected_location', dunn_pvalues)

    # Output Kruskal-Wallis p-values ​​and Dunn's test for each location
    for parameter, location_results in kruskal_p_values.items():
        print(f"Kruskal-Wallis p-values for {parameter}:")
        for location, p_value in location_results.items():
            print(f"Location: {location}, p-value: {p_value}")



def analyze_and_plot_one_graph(file_path, output_folder, groups, locations, numerical_parameters):
    # Load the data
    data = pd.read_excel(file_path)

    # Filtering data by age groups and locations 
    filtered_data = data[(data['Group'].isin(groups)) & (data['selected_location'].isin(locations))]

    # Function to calculate group positions for plotting
    def calculate_positions(start_index, num_groups, width=0.8):
        centre = start_index + width / (2 * num_groups)
        offset = width / num_groups
        positions = [centre - (num_groups / 2 - i) * offset for i in range(num_groups)]
        return positions

    # Violin plot function
    def plot_violin(data, parameter, category, hue):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=category, y=parameter, hue=hue, data=data, fill=False)
        plt.xlabel(category.replace('_', ' '))
        plt.ylabel(parameter)
        plt.legend(title=hue)
        
        sns.despine()

    # Perform statistical tests
    age_groups = groups
    results = []
    p_values_dict = {}

    for age in age_groups:
        age_data = filtered_data[filtered_data['Group'] == age]
        locations = age_data['selected_location'].unique()
        p_values_matrix = pd.DataFrame(index=locations, columns=locations, dtype=float)

        for (loc1, loc2) in combinations(locations, 2):
            group1 = age_data[age_data['selected_location'] == loc1][numerical_parameters[0]]
            group2 = age_data[age_data['selected_location'] == loc2][numerical_parameters[0]]
            stat, p_value = mannwhitneyu(group1, group2)
            results.append({'Group': age, 'Location_Comparison': f'{loc1} vs {loc2}', 'Test': 'Mann-Whitney U', 'p_value': p_value})
            p_values_matrix.loc[loc1, loc2] = p_value
            p_values_matrix.loc[loc2, loc1] = p_value

        p_values_dict[age] = p_values_matrix

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)

    # Plotting violin plots for the numerical parameters
    for parameter in numerical_parameters:
        plot_violin(filtered_data, parameter, 'Group', 'selected_location')
        for age_index, age in enumerate(age_groups):
            age_data = filtered_data[filtered_data['Group'] == age]
            locations = age_data['selected_location'].unique()
            local_param_data = age_data[parameter]
            local_p_values = p_values_dict[age]
            groups_x_pos = calculate_positions(age_index, np.size(locations))
            plot_pairwise_pvalues(local_param_data, locations, local_p_values, groups_x_pos)
        plt.show()
            
    # Save the results
    #results_df.to_excel(f'{output_folder}/statistical_results.xlsx', index=False)