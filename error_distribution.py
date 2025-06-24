def calculate_distribution_devation(error_free_results, error_prone_results):
    
    error_free_df = pd.DataFrame.from_dict(data = error_free_results[0].data.meas.get_counts(), orient='index').reset_index()
    error_free_df.columns = ['outcome', 'count']
    error_free_df['prob'] = error_free_df['count']/error_free_df['count'].sum()

    error_prone_df = pd.DataFrame.from_dict(data = error_prone_results[0].data.meas.get_counts(), orient='index').reset_index()
    error_prone_df.columns = ['outcome', 'count']
    error_prone_df['prob'] = error_prone_df['count']/error_prone_df['count'].sum()

    df_merged = error_free_df.merge(error_prone_df, on = 'outcome', how = 'outer').fillna(0)
    print(df_merged)
    return 1-(df_merged['prob_x'] - df_merged['prob_y']).abs().sum()

