"""
Data

Developers:
James Briggs

Description:
Script used for data import, manipulation, and export.
"""

import numpy as np
import pandas as pd
import re
import os
import json
import formulae as f


# set the datatypes of modelling columns
STR_COLS = ['loan_id', 'product']

DATE_COLS = ['origination_date', 'reversion_date']

NUM_COLS = ['rate_term', 'loan_amount', 'initial_rate', 'reversion_rate',
            'term', 'interest_only_amount', 'upfront_fees', 'upfront_costs',
            'entity_eir']


# define mappings class to organise column mappings
class Mappings:
    """
    Mappings class defines column mappings required for the Cashflow/EIR
    model.
    """
    def __init__(self):
        """
        Initialises an empty mappings dictionary containing all columns that
        require mapping for Cashflow/EIR modelling.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """

        # initialise the empty mappings dictionary
        self.data = {
                'loan_id': None,
                'product': None,
                'origination_date': None,
                'reversion_date': None,
                'rate_term': None,
                'loan_amount': None,
                'initial_rate': None,
                'reversion_rate': None,
                'term': None,
                'interest_only_amount': None,
                'upfront_fees': None,
                'upfront_costs': None,
                'entity_eir': None
                }
    
    
    def update(self, internal, external, allowable):
        """
        Used to update a mapping.
        
        Parameters
        ----------
        internal : str
            Internal name of the column (the mapping dictionary key).
        external : str
            External name of the column (the mapping dictionary value).
        allowable : list
            List of allowable column labels. If usins a Pandas dataframe,
            allowable values are easily extracted using 'df.columns'. This
            is used for confirming the given external label maps to something.
        
        Returns
        -------
        None.
        """

        # check the external label is contained within the allowable list
        if external in allowable:
            self.data[internal] = external
        else:
            # if not, raise an error
            raise KeyError(f"External label '{external}' does not exist in "
                           f"the allowable set of values.")


    def new(self, df):
        """
        Allows user to enter new column label mappings via a command line
        interface.

        Parameters
        ----------
        df : Pandas DataFrame object
            dataframe containing external data with original column labels.
            This is used to check that user entered columns match to external
            column labels.

        Returns
        -------
        None.
        """

        cols = df.columns  # get list of column names from dataframe
        print(cols)  # print to console for user convenience

        # loop through each key in internal mappings dictionary
        for key in self.data:
            # keep asking for the correct column name until given without typos
            while True:
                try:
                    # get external column name from user
                    ext = input(f"Type external column name giving '{key}'."
                                "\n>>> ")
                    # update mappings
                    self.update(key, ext, list(cols))
                    # break from while-loop
                    break

                except KeyError as e:
                    # if column label not recognised, ask again
                    print(f"KeyError: {e}"
                          "\nCheck for typos."
                          "\nAvailable columns are:\n"
                          f"{list(cols)}")


    def save(self, file="setup", path="settings"):
        """
        Saves the mapping dictionary to file as a JSON.
        
        Parameters
        ----------
        file : str, optional
            Filename for the mapping dictionary json.
        path : str, optional
            Local/global path to the directory that the json will be saved to.
            The default is 'settings'.
        
        Returns
        -------
        None.
        """

        # check if filename already contains file extension, if not, add it
        if file[-5:] != '.json':
            file += '.json'
        # save mappings data to file
        with open(os.path.join(path, file), 'w') as file:
            json.dump(self.data, file)


    def load(self, file="setup", path="settings"):
        """
        Loads the mapping dictionary JSON to the object's internal self.data
        variable.
        
        Parameters
        ----------
        file : str, optional
            Filename of the mapping dictionary json.
        path : str, optional
            Local/global path to the directory that the json will be loaded
            from. The default is 'settings'.
        
        Returns
        -------
        None.
        """

        # check if filename already contains file extension, if not, add it
        if file[-5:] != '.json':
            file += '.json'
        # load mappings from file
        with open(os.path.join(path, file), 'r') as file:
            self.data = json.load(file)
    
    def reverse(self):
        # outputs reversed version of self.data dictionary
        return {self.data[key]: key for key in self.data}


def output(df, path="./outputs", file="output"):
    """
    Function used to save csv's.

    Parameters
    ----------
    path : TYPE, optional
        DESCRIPTION. The default is "./outputs".
    preappend : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    """
    # if output directory does not already exist, make it
    if not os.path.isdir(path):
        os.makedirs(path)

    # check that the user has included file extension, if so, remove it
    if '.csv' in file:
        file = file.replace('.csv', '')

    # merge path and file
    full_path = os.path.join(path, f"{file}.csv")
    
    while True:
        try:
            df.to_csv(full_path, sep='|', index=False)
            # update user
            print(f"{file} data issues saved to "
                  f"'{full_path}'.")
            # if data saved successfully we break the while loop
            break
        except PermissionError:
            # user or another has file open, request to close or rename
            rename = input(f"'{full_path}' is open, please close and press <Enter>"
                           " or type a new filename (and press <Enter>).")
            if rename.strip() == '':
                pass
            elif '.csv' in rename.strip():
                full_path = os.path.join(path, rename)  # merge path and file
            else:
                full_path = os.path.join(path, rename+'.csv')
        

def format_loanbook(loanbook, mapping, conv_full_term=False, verbose=True):
    """
    Formats loanbook data into the correct format for EIR processing.

    Parameters
    ----------
    loanbook : Pandas DataFrame Object
        dataframe containing raw loanbook data.
    mapping : dictionary
        dictionary storing all column mappings in format:
            EXTERNAL: INTERNAL
    conv_full_term : Boolean, optional
        True/False value defining whether to assume non-numeric values in the
        'rate_term' column indicate full-term and therefore default to 'term'
        column value.
    verbose : Boolean, optional
        True/False indicating whether to print warnings to the console.

    Returns
    -------
    pandas dataframe
        Dataframe containing loan data.

    """
    # rename columns to internal names using mapping dictionary
    loanbook.rename({col: mapping[col] for col in loanbook.columns
                     if col in mapping},
                    axis=1, inplace=True)

    # if conv_full_term is True we assume rate_term values that are strings
    # are 'full term' and convert them to match term values / 12 (month > year)
    if conv_full_term:
        # define numeric type checking function
        def conv_rates(rate_term, term):
            if str(rate_term).isnumeric():
                return rate_term
            else:
                return float(term) / 12

        # apply numeric checking function to dataframe
        loanbook['rate_term'] = list(map(
                conv_rates, loanbook['rate_term'], loanbook['term']
                ))

    # convert from external dtype to internal dtype
    for col in loanbook.columns:
        try:
            # convert columns
            if col in STR_COLS:
                loanbook[col] = pd.Series(
                    loanbook[col].astype(str), index=loanbook.index)

            elif col in DATE_COLS:
                loanbook[col] = pd.Series(
                    pd.to_datetime(loanbook[col]), index=loanbook.index)

            elif col in NUM_COLS:
                loanbook[col] = pd.Series(
                    pd.to_numeric(loanbook[col]), index=loanbook.index)

            else:
                if verbose:
                    print(f"WARNING: '{col}' datatype not set. This column "
                          "will default to string.")
                loanbook[col] = pd.Series(
                    loanbook[col].astype(str), index=loanbook.index)
        except ValueError as e:
            raise ValueError(f"ValueError for column '{col}':" + "\n" + f"{e}")
    
    return loanbook


def calc_loanbook(loanbook, verbose=True):
    """
    Formats loanbook data into the correct format for EIR processing.

    Parameters
    ----------
    loanbook : Pandas DataFrame Object
        dataframe containing raw loanbook data.
    verbose : Boolean, optional
        True/False indicating whether to print warnings to the console.

    Returns
    -------
    pandas dataframe
        Dataframe containing loan data.

    """
    # other calculated columns
    loanbook['total_repayment'] = pd.Series(
        loanbook['loan_amount'] - loanbook['interest_only_amount'],
        index=loanbook.index)

    # amount that is repayed monthly
    loanbook['monthly_repay'] = pd.Series(-np.pmt(
        loanbook['initial_rate']/12,
        loanbook['term'],
        loanbook['total_repayment']),
            index=loanbook.index)

    # initial monthly payment IO = interest rate * IO amt / months in year
    loanbook['monthly_repay_io'] = pd.Series(
        (loanbook['initial_rate']*loanbook['interest_only_amount'])/12,
        index=loanbook.index)

    loanbook['monthly_repay_io_reversion'] = pd.Series(
        (loanbook['reversion_rate']*loanbook['interest_only_amount'])/12,
        index=loanbook.index)

    balance_on_reversion = []
    # iterate through each loan and check we have correct data and if so
    # perform calculation
    for i in range(len(loanbook)):

        cumprinc = f.princcum(
            loanbook['initial_rate'].iloc[i]/12,
            loanbook['term'].iloc[i],
            loanbook['total_repayment'].iloc[i],
            int(loanbook['rate_term'].iloc[i])*12,
            1
        )
        balance_on_reversion.append(
            loanbook['total_repayment'].iloc[i] - cumprinc)

    loanbook['reversion_balance'] = pd.Series(balance_on_reversion,
                                              index=loanbook.index)

    monthly_repay_reversion = []
    dtype_warning = 0  # initialise datatype issues counter
    # iterate through each loan and check we have correct data and if so
    # perform calculation
    for i in range(len(loanbook)):
        if loanbook['total_repayment'].iloc[i] != 0:
            pmt = (-np.pmt(
                loanbook['reversion_rate'].iloc[i] / 12,
                loanbook['term'].iloc[i] - (
                int(loanbook['rate_term'].iloc[i])*12),
                loanbook['reversion_balance'].iloc[i]))
            monthly_repay_reversion.append(pmt)
        else:
            dtype_warning += 1
            monthly_repay_reversion.append(0)

    # warn user if some rows could not be converted
    if dtype_warning != 0 and verbose:
        print(f"Warning: {dtype_warning} 'Monthly Repay Reversion' rows "
              "can not be calculated as the 'total_repayment' value is 0. "
              "These 'monthly_repay_reversion' values have defaulted to 0.")
    loanbook['monthly_repay_reversion'] = pd.Series(monthly_repay_reversion,
                                                    index=loanbook.index)

    loanbook = adjust(loanbook)  # formatting adjustment columns (if any)

    loanbook['upfront_costs'] = pd.Series(
        loanbook['upfront_costs'].fillna(0), index=loanbook.index)
    loanbook['upfront_fees'] = pd.Series(
        loanbook['upfront_fees'].fillna(0), index=loanbook.index)

    return loanbook


def format_array(array, mapping):
    """
    Formats a mostly numeric dataframe into the correct format for
    calculations. Typically, the only non-numeric column should be the product
    column.

    Parameters
    ----------
    array : Pandas DataFrame Object
        dataframe containing mostly numeric data.
    mapping : dictionary
        dictionary storing all column mappings in format:
            EXTERNAL: INTERNAL

    Returns
    -------
    pandas dataframe
        Dataframe of containing loan data.

    """

    # rename columns to internal names using mapping dictionary
    array.rename({col: mapping[col] for col in array.columns
                  if col in mapping},
                 axis=1, inplace=True)

    # convert from external dtype to internal dtype
    for col in array.columns:

        # convert columns
        if col in STR_COLS:
            array[col] = pd.Series(
                array[col].astype(str), index=array.index)

        else:
            # otherwise we assume is numeric column, eg monthly CPR or ERC
            array[col] = pd.Series(
                pd.to_numeric(array[col]), index=array.index)

    # format numeric column headers to not contain anything other than number
    array.rename({col: re.sub('\D', '', str(col)) for col in array.columns \
                  if col not in STR_COLS},
                 axis=1, inplace=True)

    # any empty cells can be replaced with 0
    array = array.fillna(0)

    return array


def search_col(sheet, start, value, limit=200):
    """Searches a column in an openpyxl sheet for a specific value. This is to
    stop the script from breaking if users add/remove rows when entering
    variables etc.

    Parameters
    ----------
    sheet : openpyxl wb sheet object
        an openpyxl sheet to be searched
    start : str
        the cell to begin searching from, eg 'C5'
    value : str (could also be int, would not recommend float)
        what value to find
    limit : int
        after iterating through this many rows, give up

    Returns
    -------
    (col, row) : tuple
        contains column (str) and row (int) where value was found
    """
    # first, split column letter from row number in start variable
    re_str = re.compile('[^a-zA-Z]')  # will identify all non letters
    col = re_str.sub('', start)  # get our column

    re_int = re.compile('[^0-9]')  # will identify all non integers
    row = int(re_int.sub('', start))  # get our row

    # start iterating through upto limit or when we find the value
    for _ in range(limit):
        if sheet[col+str(row)].value == value:
            # if we find the value, return where we found it
            return (col, row)
        # if we don't find the value, move onto the next row
        row += 1
    # if we iterate over the limit number, we assume the value cannot be found
    # and raise an error
    raise KeyError(f'{value} not found in column {col} of workbook.')


def adjust(loanbook):
    """
    Reads loanbook dataframe and extracts adjustment columns. These are
    identified by splitting on 'adjust'.

    Parameters
    ----------
    loanbook : Pandas DataFrame Object
        dataframe containing loanbook data.

    Returns
    -------
    None.

    """

    # get names of AG8 columns
    ag8_cols = [x for x in loanbook.columns if 'adjust' in x.lower()]

    # remove NaN values from AG8 columns
    for col in ag8_cols:
        loanbook[col] = pd.Series(loanbook[col].fillna(0),
                                  index=loanbook.index)

    return loanbook


def make_loans(volume):
    """
    Randomly generates a loanbook with corresponding CPR Curves and ERC Lookup.
    
    Parameters
    ----------
    volume : int
        Number of loans to randomly generate.
    
    Returns
    -------
    loanbook : Pandas DataFrame Object
        dataframe containing loanbook data
    cpr : Pandas DataFrame Object
        dataframe containin cpr curves
    erc : Pandas DataFrame Object
        dataframe containing erc profiles
    """
    # set the datatypes of modelling columns
    #STR_COLS = ['loan_id', 'product']
    
    #DATE_COLS = ['origination_date', 'reversion_date']
    
    #NUM_COLS = ['rate_term', 'loan_amount', 'initial_rate', 'reversion_rate',
    #            'term', 'interest_only_amount', 'upfront_fees', 'upfront_costs',
    #            'entity_eir']

    # initialise loanbook data
    loanbook = pd.DataFrame({'loan_id': range(volume)})

    # randomly select number of products
    num_products = round(volume / np.random.randint(2, 5))

    # create list of product names from this
    products = [f"Product {i}" for i in range(num_products)]

    # initialise

    # create random origination dates within a range
    raise ValueError("Function not built yet.")
        
