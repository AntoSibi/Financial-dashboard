#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
    

#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    
    # Add dashboard title and description
    st.title("MY FINANCIAL DASHBOARD ⭐")

    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    
    
    # Add a button to update the stock data
    Update = st.button("Update Data", key="Duh")
    

#==============================================================================
# Tab 1 - Company Profile
#==============================================================================

def render_tab1():
    """
    This function renders Tab 1 - Company Profile of the dashboard.
    """
    # Load the list of S&P 500 companies
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    # Add the selection boxes
    col1, col2, col3 = st.columns(3)  # Create 3 columns
    
    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Select Ticker", ticker_list)
    
    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col3.date_input("End date", datetime.today().date())
    
    # Display stock image
    col1 = st.columns([1, 3, 1])
   
    
    # Function to get company information from Yahoo Finance
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function gets the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
    
    # Function to get stock price data for the selected duration
    @st.cache_data
    def GetStockData(ticker, duration):
        end_date = datetime.today().date()
        if duration == '1M':
            start_date = end_date - timedelta(days=30)
        elif duration == '3M':
            start_date = end_date - timedelta(days=3 * 30)
        elif duration == '6M':
            start_date = end_date - timedelta(days=6 * 30)
        elif duration == 'YTD':
            start_date = datetime(end_date.year, 1, 1).date()
        elif duration == '1Y':
            start_date = end_date - timedelta(days=365)
        elif duration == '3Y':
            start_date = end_date - timedelta(days=3 * 365)
        elif duration == '5Y':
            start_date = end_date - timedelta(days=5 * 365)
        elif duration == 'MAX':
            start_date = None  # Fetch data from the beginning
        else:
            st.error("Invalid duration selected.")
            return None
        
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])  # Convert to datetime
        return stock_df
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        # Show the company description using markdown + HTML
        st.write('**1. Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + info['longBusinessSummary'] + '</div><br>',
                    unsafe_allow_html=True)
        
        # Show some statistics as a DataFrame
        st.write('**2. Key Statistics:**')
        info_keys = {'previousClose': 'Previous Close',
                     'open': 'Open',
                     'bid': 'Bid',
                     'ask': 'Ask',
                     'marketCap': 'Market Cap',
                     'volume': 'Volume'}
        company_stats = pd.DataFrame({'Value': pd.Series({info_keys[key]: info[key] for key in info_keys})})  # Convert to DataFrame
        st.dataframe(company_stats)
        
        # Dropdown for selecting different durations
        duration_options = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX']
        selected_duration = st.selectbox('Select Duration:', duration_options)
        
        # Get stock price data for the selected duration
        stock_price = GetStockData(ticker, selected_duration)
        
        # Plot the line graph of stock prices
        st.write('**3. Stock Price Line Graph**')
        fig, ax = plt.subplots()
        ax.plot(stock_price['Date'], stock_price['Close'], label='Closing Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'Stock Price Over Time ({selected_duration})')
        ax.legend()
        st.pyplot(fig)
    
    # Function to get major holders
    @st.cache
    def GetMajorHolders(ticker):
        for tick in ticker:
            holders = yf.Ticker(tick).major_holders
            holders = holders.rename(columns={0: "Value", 1: "Breakdown"})
            holders = holders.set_index('Breakdown')
            holders.loc[['Number of Institutions Holding Shares']].style.format(
                {'Number of Institutions Holding Shares': '{:0,.0f}'.format})
        return holders
    
    # Display major holders table
    holders = GetMajorHolders([ticker])
    st.write('**4. Major Holders**')
    st.table(holders)

    
#=============================================================================
# Tab 2 - Stock Price Chart
#==============================================================================
def render_tab2():
    # Add date range selection
    col1, col2, col3 = st.columns(3)
    start_date_chart = col1.date_input("Select Start Date for Chart", start_date)
    end_date_chart = col2.date_input("Select End Date for Chart", end_date)
    
    # Add duration and interval selection
    duration_options = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
    interval_options = ["1d", "1wk", "1mo"]
    duration = st.selectbox("Select Duration", duration_options)
    interval = st.selectbox("Select Interval", interval_options)
    
    # Add plot type selection
    plot_type = st.radio("Select Plot Type", ["Line Plot", "Candle Plot"])
    
    # Add simple moving average (SMA) checkbox
    show_sma = st.checkbox("Show Simple Moving Average (SMA)")
    
    # Add stock price retrieval
    @st.cache_data
    def GetStockData(ticker, start_date, end_date, duration, interval):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        if duration != "MAX":
            stock_df = stock_df.tail(int(duration[:-1]) * 21)  # Assuming an average of 21 trading days per month
        return stock_df
    
    # Display stock price chart
    if ticker != '':
        stock_price = GetStockData(ticker, start_date_chart, end_date_chart, duration, interval)
    
    st.write('**Stock Price Chart**')
    
    if plot_type == "Line Plot":
        st.line_chart(stock_price.set_index('Date')['Close'])
    
    elif plot_type == "Candle Plot":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=['Candlestick Chart', 'Volume'])
        
        # Candlestick Chart
        fig.add_trace(go.Candlestick(
            x=stock_price['Date'],
            open=stock_price['Open'],
            high=stock_price['High'],
            low=stock_price['Low'],
            close=stock_price['Close'],
            name='Candlestick'
        ), row=1, col=1)
        
        # SMA (Simple Moving Average)
        if show_sma:
            stock_price['SMA'] = stock_price['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=stock_price['Date'],
                y=stock_price['SMA'],
                mode='lines',
                name='SMA (50)'
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=stock_price['Date'],
            y=stock_price['Volume'],
            name='Volume',
            marker_color='blue'
        ), row=2, col=1)
        
        # Update x-axis format
        fig.update_xaxes(type='category', tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)

        
#=============================================================================
# Tab 3
#==============================================================================
def render_tab3():
    # Dashboard title and description
    st.header('Financials')

    # Columns to organize better the parameters
    col1, col2 = st.columns([25, 25])
    
    # Inside the created form, a selectbox with the type of statement.
    with col1:
        select_financial = st.selectbox("Select the type of financial information", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
    
    # And the time window of analysis
    with col2:
        select_time_financial = st.radio('', ['Annual', 'Quarterly'], key='select_time_financial')
    
    # Create button to show results.
    if st.button('Show statements'):
        selected_ticker = st.session_state.selected_ticker
        # Function for retrieving financials.
        @st.cache
        def get_company_financials(ticker):
            # Different calls for annual and quarterly info.
            if select_financial == 'Balance Sheet':
                return yf.Ticker(ticker).balance_sheet if select_time_financial == 'Annual' else yf.Ticker(ticker).quarterly_balance_sheet
            elif select_financial == 'Income Statement':
                return yf.Ticker(ticker).financials if select_time_financial == 'Annual' else yf.Ticker(ticker).quarterly_financials
            else:
                return yf.Ticker(ticker).cashflow if select_time_financial == 'Annual' else yf.Ticker(ticker).quarterly_cashflow

        # Fill nulls with $0, change format of columns, and restyle the format of cells.
        financial_data = get_company_financials(selected_ticker).fillna(0).rename(
            lambda t: 'Up to: ' + t.strftime('%d-%B-%Y'), axis='columns').style.format("${:0,.2f}")

        # Show the name of the selected info and the table
        st.header(f'{select_time_financial} {select_financial} for {selected_ticker}')
        st.table(financial_data)

#=============================================================================
# Tab 4
#==============================================================================
# Function to perform Monte Carlo simulation
@st.cache_data
def montecarlo_simulation(tickers, n_simulations, time_period):
    # Prediction based on the last 30 days' closing price
    for ticker in tickers:
        stock_name = yf.Ticker(ticker)
        stock_price = stock_name.history(period='1mo')
        close_price = stock_price['Close']
        
        # Calculating daily volatility of the stock
        daily_return = close_price.pct_change()
        daily_volatility = np.std(daily_return)
        
        # Running the simulation
        simulation_df = pd.DataFrame()
        for i in range(n_simulations):        
            # List to store the next stock price
            next_price = []
            
            # Creating the next stock price
            last_price = close_price[-1]
            for x in range(time_period):
                # Generate random percentage change around mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generating random future price
                future_price = last_price * (1 + future_return)

                # Saving the price and going to the next
                next_price.append(future_price)
                last_price = future_price
            
            # Storing the result of the simulation
            simulation_df[i] = next_price
        
        return simulation_df

# Function to fetch ticker list from S&P 500 companies
def fetch_ticker_list():
    return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Fetching ticker list from S&P 500 companies
ticker_list = fetch_ticker_list()

# Function to select a ticker from the list
def select_ticker():
    tk = "unique"
    return st.selectbox("Select a ticker", ticker_list, index=45, key=tk)

# Function to plot Monte Carlo simulation results
def plot_montecarlo_results(mc_sim, close_price, time_period):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(mc_sim)
    plt.title('Monte Carlo simulation for stock' + ' in the next ' + str(time_period) + ' days')
    plt.xlabel('Day')
    plt.ylabel('Price')

    # Highlighting current stock price
    plt.axhline(y=close_price[-1], color='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
    
    return fig

# Function to plot Value at Risk (VaR) visualization
def plot_var_visualization(ending_price):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(ending_price, bins=50)
    
    # Highlighting 5th percentile of the future price
    plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
    plt.legend(['5th Percentile of the Future Price: ' + str(np.round(np.percentile(ending_price, 5), 2))])
    plt.title('Distribution of the Ending Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    return fig

# Function to calculate and display Value at Risk (VaR)
def calculate_and_display_var(close_price, ending_price):
    future_price_95ci = np.percentile(ending_price, 5)
    VaR = close_price[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

# Main function to render Tab4
def render_tab4():
    # Number of simulations and time horizon selection
    n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
    time_period = st.selectbox("Time Horizon", [30, 60, 90])
    
    # Selecting a ticker from the list
    ticker = select_ticker()
    
    # If a valid ticker is selected, perform Monte Carlo simulation
    if ticker != '-':
        # Perform Monte Carlo simulation
        mc_sim = montecarlo_simulation([ticker], n_simulations, time_period)
        
        # Fetching last 30 days OHLC data and save close price
        stock_name = yf.Ticker(ticker)
        stock_price = stock_name.history(period='1mo')
        close_price = stock_price['Close']
        
        # Plotting Monte Carlo simulation results
        fig = plot_montecarlo_results(mc_sim, close_price, time_period)
        st.pyplot(fig)
        
        # Valuing at Risk calculation and visualization
        st.subheader('Value at Risk (VaR)')
        ending_price = mc_sim.iloc[-1:, :].values[0, ]
        fig1 = plot_var_visualization(ending_price)
        st.pyplot(fig1)

        # Displaying Value at Risk (VaR) at 95% confidence interval
        calculate_and_display_var(close_price, ending_price)
        
#=============================================================================
# Tab 5
#==============================================================================
def render_tab5():
    # Check if ticker is selected
    if ticker != '':
        stock_name = yf.Ticker(ticker)
        news = stock_name.news

        if news:
            for i in news:
                st.write(f'{i["title"]}\n{i["link"]}')
        else:
            st.write("No news available for the selected ticker.")
    else:
        st.warning("Please select a ticker in Tab 1 before checking news.")

 
#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Company profile", "Chart", "Financials", "Monte Carlo Simulation", "News"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    # Set the selected_ticker in the session_state with a dropdown list
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    st.session_state.selected_ticker = st.selectbox("Select Company", sp500_tickers)
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
