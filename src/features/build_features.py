import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:

    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)


    # These exact mappings are hardcoded in serving pipeline
    
    # Yes or No mapping 
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    # Gender mapping 
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")
    
    # For any other 2 category feature use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:

    df = df.copy()

    # Find categorical columns and excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Binary features, get binary encoding
    # Multi-category, get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    

    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))


    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    #One-Hot Encoding for Multi-Category 
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True 
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)

    # Data Type Cleanup

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df