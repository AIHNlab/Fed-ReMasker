def impute_mean_file(dfs_dict_norm, args, global_stats=None):
    dfs_dict_imputed = {}
    for name, df in dfs_dict_norm.items():
        df_imputed = df.copy()

        for col in df.columns:
            if not df[col].isna().any():
                continue
            if global_stats[col]["categorical"]:
                fill_val = global_stats[col]["mode"]
            else:
                fill_val = (global_stats[col]["mean"] - global_stats[col]["min"]) / global_stats[col]["scale"]
            df_imputed[col] = df_imputed[col].fillna(fill_val)
        dfs_dict_imputed[name] = df_imputed
        
    return dfs_dict_imputed
