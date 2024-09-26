import pandas as pd
import pytest
from pipelines.pre_process import validate_columns  

def test_validate_columns_all_present():
    # Test case where all required columns are present
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
    required_columns = ['col1', 'col2']
    
    # This should not raise any exception
    validate_columns(df, required_columns)

def test_validate_columns_missing_column():
    # Test case where a required column is missing
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    required_columns = ['col1', 'col3']
    
    # Expect a ValueError because 'col3' is missing
    with pytest.raises(ValueError, match="Missing required column: col3"):
        validate_columns(df, required_columns)

def test_validate_columns_extra_columns():
    # Test case where DataFrame has extra columns, but still includes required ones
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6], 'col4': [7, 8]})
    required_columns = ['col1', 'col2']
    
    # This should not raise any exception because required columns are present
    validate_columns(df, required_columns)
