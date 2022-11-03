import pandas as pd


def BasicEda(df: pd.DataFrame, name: str, components: list = None) -> None:
    """
    Prints the basic summary of the DataFrame, including the shape, percentage
    of null and duplicate rows, and the numeric/categorical split.

    Parameters
    ----------
    df: The DataFrame on which the EDA will be performed.
    name: The name that we want to give to the given DataFrame.
    components: {None, 'shape', 'null', 'duplicates', 'columns', 'dtypes'}, default None. 
    List that determines which components of the EDA to print. Mulitple values can be put in the list. 
    - ``shape``: Prints the rows and columns of the DataFrame.
    - ``null``: Prints the null row count and percentage.
    - ``duplicates``: Prints the duplicate row count and percentage.
    - ``columns``: Prints the dtype of the columns in the DataFrame.
    - ``dtypes``: Prints the categorical and numerical column count.

    Returns
    -------
    None
    """
    title = name.upper()
    rows = df.shape[0]
    columns = df.shape[1]
    null_rows = len(df[df.isna().all(axis=1)])
    percentage_null_rows = null_rows / rows
    types = df.dtypes
    num_cols = len(df.select_dtypes('number').columns)
    cat_cols = len(df.select_dtypes('object').columns)

    print(f'{title}', '-' * len(title), sep='\n', end='\n\n')

    if (not components) or ('shape' in components):
        print(f'Rows: {rows}    Columns: {columns}', end='\n\n')
    if (not components) or ('null' in components):
        print(f'Total null rows: {null_rows}')
        print(f'Percentage null rows: {percentage_null_rows: .3f}%', end='\n\n')
    if (not components) or ('duplicates' in components):
        duplicate_rows = df.duplicated().sum()
        percentage_duplicate_rows = duplicate_rows / rows
        print(f'Total duplicate rows: {duplicate_rows}')
        print(f'Percentage duplicate rows: {percentage_duplicate_rows: .3f}%', end='\n\n')
    if (not components) or ('columns' in components):
        print(types, end='\n\n')
    if (not components) or ('dtypes' in components):
        print(f'Number of categorical columns: {cat_cols}')
        print(f'Number of numeric columns: {num_cols}')