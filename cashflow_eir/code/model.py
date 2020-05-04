"""
Model

Developers:
James Briggs

Description:
Module covering key computational functions for loan cashflows. The majority of
this script is the Cashflow class, which contains the cashflow calculation
itself in a method for calculating NPV, EIR, and P&L from the calculated
cashflows, in addition to other key methods required for outputting and
visualising data.
"""


import numpy as np
import pandas as pd
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import formulae as f
import data as d


def month_diff(x, y):
    """
    Returns difference in months between two datetime objects. Expects
    y to be greater than x.

    Parameters
    ----------
    x : datetime
        The earlier datetime.
    y : datetime
        The later datetime.

    Returns
    -------
    int
        Integer giving number of months difference.

    """
    # get the difference in years * 12 + difference in months
    return (y.year - x.year) * 12 + y.month - x.month


def adjustment_month_diff(adjust_str, start_date):
    """
    Returns to relative month difference between adjust_date and start_date.

    Parameters
    ----------
    adjust_str : str
        Adjustment column header in format 'adjust %b-%y', for example
        'adjust Jun-19'.
    start_date : datetime
        Start date datetime object.

    Returns
    -------
    relative : int
        Relative difference between adjust_date and start_date in months.
    """

    # convert adjustment column header into datetime
    adjust_dt = datetime.strptime(adjust_str.lower().replace('adjust', '').strip(),
                               '%b-%y')

    # get difference in months between start_date and adjustment date
    return (adjust_dt.year - start_date.year) * 12 + \
            adjust_dt.month - start_date.month


def get_list(out, parameters):
    if out == 'all':
        # if the user wants to visualise all parameters, we build a list
        # containing pointers to all parameter arrays
        array_list = [
            parameters[key] for key in parameters
            ]
        parameter_list = [key for key in parameters]
    elif type(out) is list:
        array_list = []  # initialise list of items to plot
        parameter_list = []
        for param in out:
            param = str(param).strip().lower()
            try:
                # if the user has given their own list, iterate through it
                # and where possible append a selected array to plot_list
                array_list.append(parameters[param])
                parameter_list.append(param)
            except KeyError:
                print(f"Warning: '{param}' not added as "
                      "it does not exist. Check for typos and ensure name "
                      "matches to given options in documentation.\n"
                      "See 'get_list' in 'documentation/calculate' "
                      "for more information.")
    else:
        # if incorrect parameter passed, raise TypeError to user
        raise TypeError("'out' parameter must be either 'all' "
                        "to output all parameters, or a list "
                        "containing the names of parameters to "
                        "output.\n"
                        "See 'get_list' in 'documentation/calculate' "
                        "for more information.")

    # return the list of parameters (strings) and list of arrays
    return parameter_list, array_list

class Cashflow:
    """
    Class used to control cashflow calculation. This contains calculations in
    addition to visualisation 'plot' and result extraction 'output' methods.
    Before this is run the Loanbook, CPR Curves, and ERC Lookup tables must
    have all been formatted into the correct formats using the data script.
    """
    def __init__(self, loanbook, erc_lookup):
        """
        Initialises key parameters and arrays for cashflow calculation

        Parameters
        ----------
        loanbook : Pandas DataFrame
            Formatted loanbook data.
        erc_lookup : Pandas DataFrame
            Formatted ERC Lookup data.
        period_start : datetime
            Datetime object giving the month and year of the period start used
            in NPV calculations.
        period_end : datetime
            Datetime object giving the month and year of the period end used
            in NPV calculations.

        Returns
        -------
        None.

        """

        # get number of loans (needed for array shape)
        loans = loanbook.values.shape[0]
        # get maximum number of months (needed for array shape...
        self.m_max = erc_lookup.shape[1] - 1  # ...and calculation loop)

        # get list of all products
        self.products = loanbook['product'].str.strip().str.lower().values
        # initialise empty array for new erc_lookup format
        self.erc_lookup = np.zeros((loans, self.m_max))

        for i in range(loans):
            # build new erc_lookup array where rows match to rows in loanbook
            self.erc_lookup[i, :] = erc_lookup.drop(['product'], axis=1)[
                erc_lookup['product'].str.strip().str.lower() \
                    == self.products[i]
                ].values

        # initialise all calculation arrays as zeros
        # we will then iterate over each array calculate month-by-month
        self.early_repayment = np.zeros((loans, self.m_max))  # epmt
        self.scheduled_payment = np.zeros((loans, self.m_max))  # spmt
        self.statement_interest = np.zeros((loans, self.m_max))  # sint
        self.cumulative_amortisation = np.zeros((loans, self.m_max))  # cam
        self.cumulative_payment = np.zeros((loans, self.m_max))  # cpy
        #self.net_present_value = np.zeros((loans, self.m_max))  # npv
        self.profit_and_loss = np.zeros((loans, self.m_max))  #pls
        self.cashflow = np.zeros((loans, self.m_max))  # cashflow
        self.early_repayment_charge = np.zeros((loans, self.m_max))  # erc
        self.statement_amount = np.zeros((loans, self.m_max))  # cstmt / ostmt

        # some values we already know, so now we input these into our arrays
        # initial statement amount in month 0 is simply the loan_amount
        self.statement_amount[:, 0] = loanbook['loan_amount'].values.T

        # initialise adjustments and interest rate arrays with zeros
        self.adjustments = np.zeros((loans, self.m_max))
        self.rate = np.zeros((loans, self.m_max))
        # for the adjustments array, we must enter the adjustment amount in the
        # correct month (column) and correct loan (row), so first we build a
        # list of tuples in the format: (loan, month, amount)
        adjust = []
        # at the same time, we will build up our rate array row-by-row, this
        # is necessary as each loan (row) may have different reversion dates,
        # now we get the reversion months array
        self.reversion = np.array(
            [month_diff(loanbook.iloc[i]['origination_date'],
                        loanbook.iloc[i]['reversion_date']) \
             for i in range(len(loanbook))]
            )
        # now we iterate through each loan product
        for i in range(loans):
            # pull out a specific loan
            loan = loanbook.iloc[i]

            # find adjustment columns
            adjust_amounts = [loan[col] for col in loan.index \
                              if 'adjust' in col.lower()]
            
            # if there are adjustment columns, add (loan, month, amount)
            # tuple to list
            if len(adjust_amounts) > 0:
                adjust.extend(
                    [(i,
                      adjustment_month_diff(col, loan['origination_date']),
                      loan[col]) \
                     for col in loan.index if 'adjust' in col.lower()]
                    )
            # run through rate logic and additions to array
            if self.reversion[i] < self.m_max:
                # build Initial Rate array
                init_rate = np.array(
                    [loan['initial_rate']]*self.reversion[i]
                    )
                # build Reversion Rate array
                rev_rate = np.array(
                    [loan['reversion_rate']]*(self.m_max-self.reversion[i])
                    )
                # concatenate initial rate and reversion rates along column axis
                self.rate[i, :] = np.concatenate((init_rate, rev_rate))
            else:
                # just initial rate throughout so build full array with this value
                self.rate[i, :] = np.array(
                    [loanbook.iloc[i]['initial_rate']]*self.m_max
                    ).T
        # the adjustments array can now be built using the adjust tuple list
        for i, month, amount in adjust:
            self.adjustments[i, month] = amount

        # initialise initial costs, fees and loan amount arrays
        self.upfront_costs = np.zeros((loans, self.m_max))
        self.upfront_fees = np.zeros((loans, self.m_max))
        self.loan_amount = np.zeros((loans, self.m_max))

        # initial costs/fees only occur in month 0
        self.upfront_costs[:, 0] = loanbook['upfront_costs'].values.T
        self.upfront_fees[:, 0] = loanbook['upfront_fees'].values.T
        self.loan_amount[:, 0] = loanbook['loan_amount'].values.T

        # keep loanbook in object for outputting to file in 'output' method
        self.loanbook = loanbook

        # define our parameter mapping dictionary, mapping user given strings
        # to arrays calculated by calculate_cashflow
        self.parameter_mapping = {
            'statement interest': self.statement_interest,
            'cumulative amortisation': self.cumulative_amortisation,
            'cumulative payment': self.cumulative_payment,
            'scheduled payment': self.scheduled_payment,
            'early repayment': self.early_repayment,
            'early repayment charge': self.early_repayment_charge,
            'cashflow': self.cashflow,
            'statement amount': self.statement_amount,
            'adjustments': self.adjustments,
            'interest rate': self.rate
            }

    def calculate_cashflow(self, cpr):
        """
        Method used to run the calculations. This will iteratively calculate
        scheduled/early repayments, statement amount and interest, cumulative
        amortisation and payment, early repayment charges, cashflow, and
        profit and loss; month-by-month.

        Parameters
        ----------
        cpr : Pandas DataFrame
            Formatted CPR Curves data.

        Returns
        -------
        None.

        """
        # initialise empty array for new cpr curves format
        self.cpr = np.zeros((self.erc_lookup.shape[0], self.m_max))

        for i in range(len(self.products)):
            # build new cpr array where rows match to rows in loanbook
            self.cpr[i, :] = cpr.drop(['product'], axis=1)[
                cpr['product'].str.strip().str.lower() \
                    == self.products[i]
                ].values

        # we calculate values for all loans month-by-month
        # for most calculations we will use a mix of previous month values
        # [:, m-1] and current month values [:, m]
        for m in range(1, self.m_max):
            # here we use vectorised implementation of scheduled_payment
            # calculation from formulae.py
            self.scheduled_payment[:, m] = f.v_scheduled_payment(
                m,
                self.loan_amount[:, 0],
                self.reversion,
                self.cumulative_payment[:, m-1],
                self.statement_amount[:, m-1],
                self.early_repayment[:, m-1],
                self.scheduled_payment[:, m-1],
                self.loanbook['monthly_repay_io'].values,
                self.loanbook['monthly_repay'].values,
                self.loanbook['monthly_repay_reversion'].values,
                self.loanbook['monthly_repay_io_reversion'].values
                )

            # here we calculate the monthly statement interest, which is:
            # current month interest rate * previous month statement amount
            # (eg current month opening balance) and divide by 12 as we are
            # using annual interest rate to calculate monthly increase
            self.statement_interest[:, m] = self.rate[:, m] * \
                self.statement_amount[:, m-1] / 12

            # here we calculate the total amortisation upto this current month
            # this is just the sum of the previous month's [statement
            # interest, scheduled payment, and cumulative amortisation]
            self.cumulative_amortisation[:, m] = \
                self.statement_interest[:, m-1] + \
                self.scheduled_payment[:, m-1] + \
                self.cumulative_amortisation[:, m-1]

            # cumulative payment is (initial Loan Amount + current amortisation)
            # multiplied by the difference in current and previous months CPR
            # (eg the % amount of the loan paid off this month according to
            # CPR curves), previous cumulative payment is also added
            self.cumulative_payment[:, m] = f.cum_prepayment(
                self.cumulative_payment[:, m-1],
                self.loan_amount[:, 0],
                self.cumulative_amortisation[:, m],
                self.cpr[:, m],
                self.cpr[:, m-1]
                )

            # early repayment is calculated using the vectorised implementation
            # of the early_repayment calculation from formulae.py
            self.early_repayment[:, m] = f.v_early_repayment(
                self.statement_amount[:, m-1],
                self.statement_interest[:, m],
                self.scheduled_payment[:, m],
                self.cumulative_payment[:, m-1],
                self.cumulative_payment[:, m]
                )

            # the erc is calculated as this month's early repayment amount
            # multiplied by this months ERC % given by the erc_lookup table
            self.early_repayment_charge[:, m] = self.erc_lookup[:, m] * self.early_repayment[:, m]

            # this month's cashflow is simply the sum of all previous month's
            # payments [scheduled_payment, early_repayment] + thiis month's
            # charges and adjustments [early_repayment_charge, adjustments]
            # note: loan_amount, upfront_costs, and upfront_fees only contain
            # amounts in month 0, and thus make no impact after the first month
            self.cashflow[:, m] = f.cashflow_calc(
                self.loan_amount[:, m-1],
                self.upfront_costs[:, m-1],
                self.upfront_fees[:, m-1],
                self.scheduled_payment[:, m-1],
                self.early_repayment[:, m-1],
                self.early_repayment_charge[:, m],
                self.adjustments[:, m]
                )

            # cumulative profit and loss is simply previous month P&L - current
            # month cashflow
            self.profit_and_loss[:, m] = self.profit_and_loss[:, m-1] - self.cashflow[:, m]

            # statement amount is simply the sum of the previous month's
            # statement amount and the current month's [statement_interest,
            # scheduled_payment, early_repayment]
            self.statement_amount[:, m] = \
                self.statement_amount[:, m-1] + \
                self.statement_interest[:, m] + \
                self.scheduled_payment[:, m] + \
                self.early_repayment[:, m]


    def calculate_vals(self, period_start, period_end):
        """
        Used for calculating effective interest rate, net present value, and
        profit and loss for cashflows calculated with the cashflow method.
        
        Parameters
        ----------
        period_start : datetime
            Start date for NPV and P&L calculations.
        peiod_end : datetime
            End date for P&L calculation.
        
        Returns
        -------
        None.
        """
        
        # intialise EIR, NPV, and P&L lists
        self.eir = []
        self.npv = {}
        self.npv['calculated'] = []
        self.npv['entity'] = []
        self.pl = []

        # loop through each loan and calculate values
        for i in range(self.cashflow.shape[0]):
            # we need to know the relative months for the portfolio
            # period_start and period_end for each loan based on the loan's
            # origination_date
            start = month_diff(
                    self.loanbook.iloc[i]['origination_date'],
                    period_start
                    )
            end = month_diff(
                    self.loanbook.iloc[i]['origination_date'],
                    period_end
                    )
            
            # we only want cashflows occuring after the period_start for NPV
            npv_cashflow = [0.] + self.cashflow[i, start:]

            # calculate the EIR, we use the numpy IRR function which gives
            # us EIR as we have already taken into account interest, adjustments
            # etc - :-1 gives us the final cashflow values only as we have
            # calculated cumulative cashflow
            self.eir.append(np.irr(self.cashflow[i, :-1]))

            # we calculate the NPV with our own calculated EIR
            self.npv['calculated'].append(-np.npv(self.eir[i], npv_cashflow))
            # we calculate entity NPV using given loanbook EIR values
            self.npv['entity'].append(-np.npv(self.entity_eir[i], npv_cashflow))

            # finally, take sum of profit_and_loss per loan
            self.pl.append(sum(self.profit_and_loss[i, start:end]))


    def plot(self, products='all', out='all',
             save=False, path='./Outputs/Cashflow/Visualisation',
             limit=30):
        """
        Method for plotting key parameters onto a Matplotlib subplots object.
        These key parameters are the arrays used during the cashflow calculation.

        A maximum of five plots will be displayed before the method ends,
        unless the 'save' parameter is True, in which case product
        visualisations will be saved to file as JPEGs. Note that saving to file
        can take a long time for large numbers of files, a limit of products
        to be visualised is set by the 'limit' parameter.

        Parameters
        ----------
        products : list, optional
            List of product names identifying which products are to be
            visualised.
            The default is 'all'.
        out : list, optional
            List of parameter names identifying which parameters are to be
            visualised. The default is 'all'.
        save : Boolean, optional
            True/False value defining whether to save the plots (True) to the
            path given by 'path', or to simply display them (False).
            The default is False.
        path : str, optional
            If the 'save' parameter is True then visualisations will be saved
            to the path defined here.
            The default is './Outputs/Cashflow/Visualisation'.
        limit : int, optional
            If the 'save' parameter is True then the number of visualisations
            to be saved to file will be limited by this parameter.
            The default is 30.

        Returns
        -------
        None.

        """

        # if output directory does not already exist and save=True,
        # make the directory
        if not os.path.isdir(path) and save:
            os.makedirs(path)

        # get list of parameter names and respective arrays to visualise
        parameter_list, plot_list = get_list(out, self.parameter_mapping)

        if products == 'all':
            # if all products chosen, we do not need to select specific rows
            products = (self.loanbook['loan_id'].astype(str) +
                        self.loanbook['product'].astype(str)).values
            pass
        elif type(products) is list:
            # strip whitespace and lowercase all entries in products list
            products = [str(prod).strip().lower() for prod in products]
            # get indexes of all rows that match to a product from list
            idx = self.loanbook.index[
                self.loanbook['product'].str.strip().str.lower().isin(products)
                ]
            # specify indexes (required for indexing numpy arrays like this)
            row_idx = np.array(idx)
            # iterate through every array and only return the rows we want
            # from the idx
            plot_list = [
                array[row_idx, :] for array in plot_list
                ]
            products = self.loanbook[['product', 'loan_id']].iloc[idx]
            products = (products['loan_id'].astype(str) +
                        products['product'].astype(str)).values
        else:
            # if incorrect parameter passed, raise TypeError to user
            raise TypeError("'products' parameter must be either 'all' "
                            "to display all products, or a list "
                            "containing the names of products to "
                            "visualise.\n"
                            "See 'plot' in 'documentation/calculation/Cashflow' "
                            "for more information.")

        # we do some calculations to find optimal width and height of subplot
        width = sorted([(len(plot_list) % x, x) for x in range(2, 7)])[0][1]
        height = math.ceil(len(plot_list) / width)

        # now we loop through each product specified and create a subplot for
        # all, a maxiumum of 5 plots will be shown in the window - the save
        # parameter must be True to save plots to file
        for i in range(len(products)):
            sns.set_style('white')

            # initialise subplot object
            fig, ax = plt.subplots(nrows=height, ncols=width,
                                   figsize=(5*width, 4*height))
            # initialise x and y position of subplot
            x = 0
            y = 0
            # create data dictionary to be looped through
            data_dict = {
                parameter_list[j]: plot_list[j][i] \
                    for j in range(len(parameter_list))
                }

            for param in data_dict:
                # careful with indexing here, if height is 1 there is no
                # second index (eg y) to index
                if height == 1:
                    axes = ax[x]
                else:
                    axes = ax[x, y]

                # plot the parameter array
                axes.plot(range(len(data_dict[param])), data_dict[param],
                              color='#726EFF', linewidth=2)
                # set the subplot title so that we know what it refers to
                axes.set_title(param)

                # iterate through each subplot position
                if x == width-1:
                    x = 0
                    y += 1
                else:
                    x += 1
            # set the plot's main title to loanbook Loan ID + Product
            fig.suptitle(products[i], fontsize=14)
            if save and i < limit-1:
                # if save is True, we save the plot to file
                plt.savefig(os.path.join(path, f"{products[i]}.jpg"))
                plt.close(fig)  # clear memory
                # also update user every 10 figures
                if i % 10 == 0:
                    print(f"Saving figure {i} '{products[i]}' to file")
            elif i < 4:
                plt.show()
                plt.close(fig)  # clear memory
            else:
                plt.close(fig)  # clear memory
                if save:
                    # update user showing where visualisations have been saved
                    print(f"{i+1} visualisations saved to {path}.")
                else:
                    # explain to user that this function allows max 5 plots
                    # to be shown via plt.show()
                    print(f"{i+1} visualisations plotted. The 'Cashflow.plot()' "
                          "function does not allow more to be actively "
                          "displayed. To view all please set 'save=True'.\n"
                          "See 'plot' in 'documentation/calculation/Cashflow' "
                          "for more information.")
                return


    def output(self, path="./Outputs/Cashflow", preappend="", vis=False,
               out='all', loanbook=False):
        """
        Method to output cashflow, calculated arrays, and loanbook with
        calculated EIR, NPV and P&L columns to CSV. Can also output array
        lineplots with 'vis' parameter.

        Parameters
        ----------
        path : str, optional
            The file path where the output files should be saved.
            The default is "./Outputs/Cashflow".
        preappend : str, optional
            Text to preappend to the filenames being saved.
            The default is "".
        vis : Boolean, optional
            True/False value defining whether to save calculated array
            visualisations to file or not.
            The default is False.
        out : str, optional
            Indicates which arrays to output. Options include:              <br>
            - 'all': outputs all of the following options                   <br>
            - 'cashflow': outputs cashflow array                            <br>
            - 'early repayment': outputs early_repayment array
            - 'scheduled payment': output scheduled_payment array           <br>
            - 'cumulative amortisation': outputs cumulatative_amortisation array<br>
            - 'early repayment charges': outputs early_repayment_charge array<br>
            - 'statement amount': outputs statement_amount array            <br>
            - 'adjustments': outputs adjustments array (note that this is
               actually an input array, and is not calculated)              <br>
            - 'interest rate': outputs rate array                           <br>
            The default is 'all'.
        loanbook : Boolean, optional
            True/False indicating whether to output input loanbook data with
            additional columnsfor calculated EIR, NPV, entity NPV, and P&L.
            The Default is False.

        Returns
        -------
        None.

        """

        # we will output key tables, all will need loan product to be added
        # and to be converted into Pandas DataFrames

        # first we create a single column dataframe containing product names
        products = pd.DataFrame({'product': self.products})

        # create cashflows dataframe
        cashflows = pd.concat([products, pd.DataFrame(self.cashflow)],
                              ignore_index=True,
                              axis=1)
        # save to file
        d.output(cashflows, path=path, file=f"{preappend}cashflows")
        del cashflows

        # create loanbook with new calculated columns
        loanbook = pd.concat([self.loanbook, pd.DataFrame({
            'calculated_eir': self.eir,
            'calculated_npv': self.npv['Calculated'],
            'entity_npv': self.npv['Entity'],
            'calculated_profit_and_loss': self.pl
            })],
                             ignore_index=True,
                             axis=1)

        # save to file
        d.output(loanbook, path=path, file=f"{preappend}loanbook")
        del loanbook

        if vis:
            self.plot(save=True, path=os.path.join(path, "Visualisation"))

    # class end
