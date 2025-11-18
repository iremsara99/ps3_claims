import hashlib
from sklearn.model_selection import GroupShuffleSplit

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

def create_sample_column(df, id_column, training_frac=0.8):
    """
    Create a deterministic train/test split based on ID column using hash.
    
    This function creates a 'sample' column with 'train' and 'test' values
    based on a stable hash of the ID column, ensuring the same ID always
    gets assigned to the same split regardless of when or where the code runs.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to add the sample column to
    id_column : str or list of str
        Name(s) of the ID column(s) to base the split on
    training_frac : float, optional
        Fraction to use for training, by default 0.8 (80% train, 20% test)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with a new 'sample' column containing 'train' or 'test'
    
    Examples
    --------
    >>> df = pd.DataFrame({'IDpol': [1, 2, 3, 4, 5]})
    >>> df = create_sample_column(df, 'IDpol', training_frac=0.8)
    >>> df['sample'].value_counts()
    train    4
    test     1
    """
    import hashlib
    
    # Convert single column name to list for consistent handling
    if isinstance(id_column, str):
        id_columns = [id_column]
    else:
        id_columns = id_column
    
    # Create a combined ID string if multiple columns are provided
    # This concatenates the values with a separator
    df['_temp_id'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    
    # Function to hash a single ID and convert to bucket (0-99)
    def hash_to_bucket(id_value):
        # Create MD5 hash of the ID string
        hash_object = hashlib.md5(str(id_value).encode())
        # Get hexadecimal representation
        hex_dig = hash_object.hexdigest()
        # Convert to integer and mod 100 to get bucket (0-99)
        bucket = int(hex_dig, 16) % 100
        return bucket
    
    # Apply hash function to get bucket for each row
    df['_temp_bucket'] = df['_temp_id'].apply(hash_to_bucket)
    
    # Assign 'train' or 'test' based on training fraction
    # If bucket < threshold (e.g., 80), assign to train; otherwise test
    threshold = int(training_frac * 100)
    df['sample'] = df['_temp_bucket'].apply(
        lambda x: 'train' if x < threshold else 'test'
    )
    
    # Clean up temporary columns
    df.drop(columns=['_temp_id', '_temp_bucket'], inplace=True)
    
    return df


def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    unique_ids = df[id_column].unique()
    id_map = {original_id : numerical_id for numerical_id, original_id in enumerate(unique_ids)}
    df["numerical_id"] = df[id_column].map(id_map)

    split_generator = GroupShuffleSplit(test_size=(1-training_frac), n_splits=1, random_state=99)
    train_ids, test_ids = next(split_generator.split(df, groups=df["numerical_id"]))
    df.loc[train_ids, "sample"] = "train"
    df["sample"].fillna("test", inplace=True)
    df.drop(columns=["numerical_id"], inplace=True)

    return df
