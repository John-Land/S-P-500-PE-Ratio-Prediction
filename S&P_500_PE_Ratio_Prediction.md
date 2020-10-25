# S&P 500 Price to Earnings Ratio Prediction

The S&P 500 is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. It is one of the most commonly followed equity indices. For more information, refer to below link: <https://en.wikipedia.org/wiki/S%26P_500_Index>

A traditional measure of whether a stock is expensive or cheap, is to use a valuation multiple. A valuation multiple relates the market value of an asset relative to a key statistic that is assumed to relate to that value. To be useful, that statistic – whether earnings, cash flow or some other measure – must bear a logical relationship to the market value observed; to be seen, in fact, as the driver of that market value. For more information about valuation multiples, refer to below link: <https://en.wikipedia.org/wiki/Valuation_using_multiples#Valuation_multiples>

The most famous valuation multiple the PE (Price to earnings) ratio. This is the ratio between the market value of all the shares outstanding for a company, divided by the company’s Net Income. It is often also defined as Price per share divided by earnings per share, which is simply dividing both the Market Capitalization and the Net income by the shares outstanding.
In this project we will try to predict the PE ratios of companies in the S&P 500, based on fundamental financial metrics that should be driving these ratios.

While working through this project, I will follow a slightly adjusted Machine Learning project check list from Aurelien Geron's book "Hands-On Machine Learning with Scikit_Learn, Keras & TensorFlow". (Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (p. 35). O'Reilly Media. Kindle Edition.)
1.	Look at the big picture
2.	Get the data
3.	Discover and visualize the data to gain insights
4.	Prepare the data for Machine Learning algorithms
5.	Select, train and fine-tune models
6.	Conclusion

As with all coding posts, the full jupyter notebook can be found in my github repo below:
<https://github.com/John-Land/S-P_500_PE_Ratio_Prediction>


## 1. Look at the big picture

Before analysing our dataset, let us first try to understand what fundamental financial information about the company should be driving the PE ratio based on Financial Theory.

The present value of any Cash Flow generating asset is the value of all future cash flows the asset will generate over its lifetime, discounted back to today with a risk adjusted interest rate.

In its simplest form, where we expect the cash flow to grow at the same rate for ever, this is given by the below perpetuity formula.

![image%201.JPG](attachment:image%201.JPG)

When trying to value a company based on its future cash flows, we can either look at it through the lenses of all capital providers (equity and debt providers) and we would use the FCFF (Free Cash Flow to the Firm) as the measure of cash flow (CF), or we could look at it purely as an Equity investor and we would use the FCFE (Free Cash Flow to Equity) as our measure of CF.

The main difference is that the FCFF is the cash flow left over after cash flow needs for operations and Investments are met, but before any debt repayments, whereas the FCFE is after debt payments (cash flow left over for equity investors).

If we want to estimate the value of Equity directly, we would therefore use the FCFE as the appropriate measure of CF.

As the Market Capitalization (MC) (numerator in the PE ratio), is the aggregate market estimation of the value of Equity, we can rewrite the above formula as follows:

![image%202.JPG](attachment:image%202.JPG)

This formula gives us some understanding of what we would expect the fundamental drivers of the PE ratio to be.
The fundamental drivers of the P/E ratio are
1. A higher expected growth rate g , all else being equal, should lead to a higher PE ratio
2. A lower reinvestment need (Capex - DA + delta WC), all else being equal, should lead to a higher PE ratio
3. Lower amount of debt needed (Net Debt repayments), all else being equal, should result in a higher PE ratio
4. A higher risk relative to the overall market, measured as beta in the denominator, should lead to a lower PE ratio

rf (risk free rate traditionally measured as the US T-Bond rate) and ERP (Equity risk premium, basically the premium that equity investor require on top of the risk free rate for investing in the overall stock market) values are the same for all stocks, therefore changes in these overall market metrics will affect all PE ratios, but these will not help differentiation between PE ratios of individual companies, as these are equal for all companies.

We will therefore try to obtain financial metrics that are related to points 1-4, as we would expect these to be partial drivers of the PE ratios seeen in the market.

## 2. Get the data
The company data was downloaded from the data provider "Finbox".
For more information, refer to below link:
<https://finbox.com>

We will first import the data and check for any missing values and some basic information.
Then we will remove outliers before splitting the data into training and testing sets, and analysing the training set in more detail.


```python
# linear algebra
import numpy as np     

# data processing
import pandas as pd    

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
```


```python
company_data = pd.read_excel("Financials - unlinked.xlsx")
```


```python
company_data.shape
```




    (531, 38)




```python
company_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Full_Ticker</th>
      <th>Name</th>
      <th>Sector</th>
      <th>Industry_GICS</th>
      <th>Index_Membership</th>
      <th>Market_Cap</th>
      <th>Net_Debt</th>
      <th>Minority_Interest</th>
      <th>EV</th>
      <th>...</th>
      <th>EV_TTM_EBITDA</th>
      <th>Market_Cap_TTM_Net_Income</th>
      <th>Market_Cap_BV_Equity</th>
      <th>Interest_Expense_TTM</th>
      <th>Interest_Expense_TTM_%_TTM_Net_Income</th>
      <th>Cash_from_Investing</th>
      <th>Cash_from_Investing_%_TTM_Net_Income</th>
      <th>Revenue_TTM</th>
      <th>Cash_from_Investing_%_TTM_Revenue</th>
      <th>Interest_Expense_TTM_%_Revenue_TTM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MSFT</td>
      <td>NASDAQGS:MSFT</td>
      <td>Microsoft Corporation</td>
      <td>Information Technology</td>
      <td>Software</td>
      <td>Dow Jones Composite Average, Dow Jones Industr...</td>
      <td>1.625176e+06</td>
      <td>-54382.0</td>
      <td>0.0</td>
      <td>1.570794e+06</td>
      <td>...</td>
      <td>24.070147</td>
      <td>36.701423</td>
      <td>13.737284</td>
      <td>89.0</td>
      <td>-0.002010</td>
      <td>-12223.0</td>
      <td>-0.276033</td>
      <td>143015.0</td>
      <td>-0.085467</td>
      <td>0.000622</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAPL</td>
      <td>NASDAQGS:AAPL</td>
      <td>Apple Inc.</td>
      <td>Information Technology</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>Dow Jones Composite Average, Dow Jones Industr...</td>
      <td>1.979619e+06</td>
      <td>-71431.0</td>
      <td>0.0</td>
      <td>1.908188e+06</td>
      <td>...</td>
      <td>24.255285</td>
      <td>33.883653</td>
      <td>27.387435</td>
      <td>1052.0</td>
      <td>-0.018006</td>
      <td>-10618.0</td>
      <td>-0.181740</td>
      <td>273857.0</td>
      <td>-0.038772</td>
      <td>0.003841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AMZN</td>
      <td>NASDAQGS:AMZN</td>
      <td>Amazon.com, Inc.</td>
      <td>Consumer Discretionary</td>
      <td>Internet &amp; Direct Marketing Retail</td>
      <td>Nasdaq 100, Nasdaq Composite, Russell 1000, Ru...</td>
      <td>1.591026e+06</td>
      <td>20010.0</td>
      <td>0.0</td>
      <td>1.611036e+06</td>
      <td>...</td>
      <td>40.708424</td>
      <td>120.715166</td>
      <td>21.579670</td>
      <td>-885.0</td>
      <td>0.067147</td>
      <td>-35307.0</td>
      <td>-2.678832</td>
      <td>321782.0</td>
      <td>-0.109723</td>
      <td>-0.002750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GOOG</td>
      <td>NASDAQGS:GOOG</td>
      <td>Alphabet Inc.</td>
      <td>Communication Services</td>
      <td>Interactive Media &amp; Services</td>
      <td>Nasdaq 100, Nasdaq Composite, Russell 1000, Ru...</td>
      <td>1.095684e+06</td>
      <td>-104937.0</td>
      <td>0.0</td>
      <td>9.907473e+05</td>
      <td>...</td>
      <td>21.521609</td>
      <td>34.746124</td>
      <td>5.284940</td>
      <td>2197.0</td>
      <td>-0.069671</td>
      <td>-23943.0</td>
      <td>-0.759276</td>
      <td>166030.0</td>
      <td>-0.144209</td>
      <td>0.013233</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GOOG.L</td>
      <td>NASDAQGS:GOOG.L</td>
      <td>Alphabet Inc.</td>
      <td>Communication Services</td>
      <td>Interactive Media &amp; Services</td>
      <td>Nasdaq 100, Nasdaq Composite, Russell 1000, Ru...</td>
      <td>1.095684e+06</td>
      <td>-104937.0</td>
      <td>0.0</td>
      <td>9.907473e+05</td>
      <td>...</td>
      <td>21.521609</td>
      <td>34.746124</td>
      <td>5.284940</td>
      <td>2197.0</td>
      <td>-0.069671</td>
      <td>-23943.0</td>
      <td>-0.759276</td>
      <td>166030.0</td>
      <td>-0.144209</td>
      <td>0.013233</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



### 2.1 Data Structure and removal of not meaningful samples

We will first remove any companies with either market capitalization <0 or Net Income <0, as for these companies, the PE ratio is not meaningful. We also filter out financials, as our measurement of debt is not applicable to Financial Companies. We also remove duplicates, as we do not want the same company to appear twice in our data.


```python
company_data = company_data[(company_data.Market_Cap > 0)]
company_data = company_data[(company_data.Net_Income > 0)]
company_data = company_data[(company_data.Sector != 'Financials')]

company_data = company_data.drop_duplicates('Name' , keep='first')
pe_data = company_data[['Sector', 
                        'Name', 
                        'Market_Cap_TTM_Net_Income', 
                        'Net_Income_Forecast_CAGR_10y',
                        'Avg_Net_Income_Margin_Forecast_10y', 
                        'Beta_5y', 
                        'Net_Debt_perc_EV'
                       ]].set_index('Name')
pe_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sector</th>
      <th>Market_Cap_TTM_Net_Income</th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Microsoft Corporation</th>
      <td>Information Technology</td>
      <td>36.701423</td>
      <td>0.130946</td>
      <td>0.342895</td>
      <td>0.923331</td>
      <td>-0.034621</td>
    </tr>
    <tr>
      <th>Apple Inc.</th>
      <td>Information Technology</td>
      <td>33.883653</td>
      <td>0.0688865</td>
      <td>0.220838</td>
      <td>1.314396</td>
      <td>-0.037434</td>
    </tr>
    <tr>
      <th>Amazon.com, Inc.</th>
      <td>Consumer Discretionary</td>
      <td>120.715166</td>
      <td>0.284554</td>
      <td>0.098539</td>
      <td>1.353006</td>
      <td>0.012421</td>
    </tr>
    <tr>
      <th>Alphabet Inc.</th>
      <td>Communication Services</td>
      <td>34.746124</td>
      <td>0.112474</td>
      <td>0.195844</td>
      <td>1.106453</td>
      <td>-0.105917</td>
    </tr>
    <tr>
      <th>Facebook, Inc.</th>
      <td>Communication Services</td>
      <td>33.685330</td>
      <td>0.162782</td>
      <td>0.313488</td>
      <td>1.295642</td>
      <td>-0.063398</td>
    </tr>
  </tbody>
</table>
</div>



Below code ensures that all numerical variables are of type numeric.


```python
pe_data['Market_Cap_TTM_Net_Income'] = pe_data['Market_Cap_TTM_Net_Income'].apply(pd.to_numeric, errors='coerce')
pe_data['Net_Income_Forecast_CAGR_10y'] = pe_data['Net_Income_Forecast_CAGR_10y'].apply(pd.to_numeric, errors='coerce')
pe_data['Avg_Net_Income_Margin_Forecast_10y'] = pe_data['Avg_Net_Income_Margin_Forecast_10y'].apply(pd.to_numeric, errors='coerce')
pe_data['Beta_5y'] = pe_data['Beta_5y'].apply(pd.to_numeric, errors='coerce')
pe_data['Net_Debt_perc_EV'] = pe_data['Net_Debt_perc_EV'].apply(pd.to_numeric, errors='coerce')
```


```python
pe_data.shape
```




    (373, 6)



After removing duplicates, companies with market capitalization <= 0, Net Income <= 0 and companies in the Finical Sector, the total sample size drops from 531 to 373.


```python
pe_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 373 entries, Microsoft Corporation to Alliance Data Systems Corporation
    Data columns (total 6 columns):
     #   Column                              Non-Null Count  Dtype  
    ---  ------                              --------------  -----  
     0   Sector                              373 non-null    object 
     1   Market_Cap_TTM_Net_Income           373 non-null    float64
     2   Net_Income_Forecast_CAGR_10y        365 non-null    float64
     3   Avg_Net_Income_Margin_Forecast_10y  373 non-null    float64
     4   Beta_5y                             373 non-null    float64
     5   Net_Debt_perc_EV                    373 non-null    float64
    dtypes: float64(5), object(1)
    memory usage: 20.4+ KB
    


```python
pe_data.isna().sum()
```




    Sector                                0
    Market_Cap_TTM_Net_Income             0
    Net_Income_Forecast_CAGR_10y          8
    Avg_Net_Income_Margin_Forecast_10y    0
    Beta_5y                               0
    Net_Debt_perc_EV                      0
    dtype: int64



The dataset is missing 8 values in the variable 'Net_Income_Forecast_CAGR_10y'.


```python
pe_data = pe_data.dropna()
```


```python
pe_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 365 entries, Microsoft Corporation to Alliance Data Systems Corporation
    Data columns (total 6 columns):
     #   Column                              Non-Null Count  Dtype  
    ---  ------                              --------------  -----  
     0   Sector                              365 non-null    object 
     1   Market_Cap_TTM_Net_Income           365 non-null    float64
     2   Net_Income_Forecast_CAGR_10y        365 non-null    float64
     3   Avg_Net_Income_Margin_Forecast_10y  365 non-null    float64
     4   Beta_5y                             365 non-null    float64
     5   Net_Debt_perc_EV                    365 non-null    float64
    dtypes: float64(5), object(1)
    memory usage: 20.0+ KB
    


```python
pe_data.isna().sum()
```




    Sector                                0
    Market_Cap_TTM_Net_Income             0
    Net_Income_Forecast_CAGR_10y          0
    Avg_Net_Income_Margin_Forecast_10y    0
    Beta_5y                               0
    Net_Debt_perc_EV                      0
    dtype: int64




```python
pe_data.shape
```




    (365, 6)



After dropping the 8 missing values, the sample size drops to 365 companies.

### 2.2 Inspection for Outliers
We will look at the individual histograms of all variables, to see if we have potential outliers, and use Gaussian mixture models to later remove these outliers.

### 2.2.1 Histograms


```python
sns.histplot(pe_data[['Market_Cap_TTM_Net_Income']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_29_1.png)
    


- Extreme outliers in higher values for PE ratio
    > Outlier removal before model training would be reasonable


```python
sns.histplot(np.log(pe_data[['Market_Cap_TTM_Net_Income']]))
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_31_1.png)
    


- Log of PE ratios still shows extreme outliers in higher values for PE ratio
- Log however, reduces the extreme values, and makes the data more Gaussian, therefore should help performance of predictions
    > Predicting log of PE instead of PE would be reasonable

    > Outlier removal before model training would be reasonable


```python
sns.histplot(pe_data[['Net_Income_Forecast_CAGR_10y']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_33_1.png)
    


- No indication of extreme outliers or unreasonable values in the growth forecast for Net Income
    > No outlier removal reasonable


```python
sns.histplot(pe_data[['Avg_Net_Income_Margin_Forecast_10y']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_35_1.png)
    


- No indication of extreme outliers or unreasonable values in the margin forecast for Net Income
    > No outlier removal reasonable


```python
sns.histplot(pe_data[['Beta_5y']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_37_1.png)
    


- No indication of extreme outliers or unreasonable values in the past 5 year beta values
    > No outlier removal reasonable


```python
sns.histplot(pe_data[['Net_Debt_perc_EV']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_39_1.png)
    


- Extreme outliers in higher values for the Net Debt as % of Enterprise Value.
    > Outlier removal before model training would be reasonable

Based on the histograms for the individual variables, we conclude that there are extreme outliers in the PE ratio variable and Net debt % EV variable.

### 2.2.2 Outlier Removal with Gaussian Mixtures

Below code uses the GaussianMixture model to remove the most extreme 5% of samples, defined as the 5% of samples with the lowest density values.


```python
from sklearn.mixture import GaussianMixture
pe_ratios_train = pe_data[['Market_Cap_TTM_Net_Income']]

gm = GaussianMixture(n_components=1)
gm.fit(pe_ratios_train,)

densities = gm.score_samples(pe_ratios_train)
threshold = np.percentile(densities, 5)
anomalies_train = pe_ratios_train[densities < threshold]
anomalies_train.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Market_Cap_TTM_Net_Income</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fidelity National Information Services, Inc.</th>
      <td>2852.57</td>
    </tr>
    <tr>
      <th>JD.com, Inc.</th>
      <td>5.83</td>
    </tr>
    <tr>
      <th>Biogen Inc.</th>
      <td>8.08</td>
    </tr>
    <tr>
      <th>NetEase, Inc.</th>
      <td>2.53</td>
    </tr>
    <tr>
      <th>Sempra Energy</th>
      <td>8.69</td>
    </tr>
    <tr>
      <th>Kinder Morgan, Inc.</th>
      <td>240.45</td>
    </tr>
    <tr>
      <th>eBay Inc.</th>
      <td>7.28</td>
    </tr>
    <tr>
      <th>Sysco Corporation</th>
      <td>154.07</td>
    </tr>
    <tr>
      <th>IQVIA Holdings Inc.</th>
      <td>184.51</td>
    </tr>
    <tr>
      <th>Zimmer Biomet Holdings, Inc.</th>
      <td>807.04</td>
    </tr>
    <tr>
      <th>Chipotle Mexican Grill, Inc.</th>
      <td>153.36</td>
    </tr>
    <tr>
      <th>The Williams Companies, Inc.</th>
      <td>188.15</td>
    </tr>
    <tr>
      <th>Hilton Worldwide Holdings Inc.</th>
      <td>513.24</td>
    </tr>
    <tr>
      <th>Hewlett Packard Enterprise Company</th>
      <td>12169.19</td>
    </tr>
    <tr>
      <th>NortonLifeLock Inc.</th>
      <td>3.07</td>
    </tr>
    <tr>
      <th>NRG Energy, Inc.</th>
      <td>1.93</td>
    </tr>
    <tr>
      <th>MGM Resorts International</th>
      <td>5.80</td>
    </tr>
    <tr>
      <th>Kimco Realty Corporation</th>
      <td>4.81</td>
    </tr>
    <tr>
      <th>Xerox Holdings Corporation</th>
      <td>3.98</td>
    </tr>
  </tbody>
</table>
</div>



The GaussianMixture model detected some very low PE values, and on the upper side as expected by the initial histogram on PE ratios, some extreme values of PE ratios such as HP with a PE of 12169 and Fidelity with a PE ratio of 2852.

Below code removes the list of extreme values above from our sample.


```python
pe_data_new = pe_data[densities >= threshold]
```


```python
from sklearn.mixture import GaussianMixture
pe_ratios_train = pe_data_new[['Net_Debt_perc_EV']]

gm = GaussianMixture(n_components=1)
gm.fit(pe_ratios_train,)

densities = gm.score_samples(pe_ratios_train)
threshold = np.percentile(densities, 5)
anomalies_train = pe_ratios_train[densities < threshold]
anomalies_train.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net_Debt_perc_EV</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Walgreens Boots Alliance, Inc.</th>
      <td>0.552</td>
    </tr>
    <tr>
      <th>Baidu, Inc.</th>
      <td>3.312</td>
    </tr>
    <tr>
      <th>General Motors Company</th>
      <td>0.628</td>
    </tr>
    <tr>
      <th>FirstEnergy Corp.</th>
      <td>0.552</td>
    </tr>
    <tr>
      <th>Simon Property Group, Inc.</th>
      <td>0.541</td>
    </tr>
    <tr>
      <th>PPL Corporation</th>
      <td>0.511</td>
    </tr>
    <tr>
      <th>Arista Networks, Inc.</th>
      <td>-0.200</td>
    </tr>
    <tr>
      <th>Garmin Ltd.</th>
      <td>-0.164</td>
    </tr>
    <tr>
      <th>ONEOK, Inc.</th>
      <td>0.502</td>
    </tr>
    <tr>
      <th>CarMax, Inc.</th>
      <td>0.502</td>
    </tr>
    <tr>
      <th>ViacomCBS Inc.</th>
      <td>0.508</td>
    </tr>
    <tr>
      <th>Mylan N.V.</th>
      <td>0.597</td>
    </tr>
    <tr>
      <th>WestRock Company</th>
      <td>0.499</td>
    </tr>
    <tr>
      <th>The AES Corporation</th>
      <td>0.553</td>
    </tr>
    <tr>
      <th>Iron Mountain Incorporated</th>
      <td>0.573</td>
    </tr>
    <tr>
      <th>SL Green Realty Corp.</th>
      <td>0.586</td>
    </tr>
    <tr>
      <th>Harley-Davidson, Inc.</th>
      <td>0.688</td>
    </tr>
    <tr>
      <th>Alliance Data Systems Corporation</th>
      <td>0.862</td>
    </tr>
  </tbody>
</table>
</div>



The GaussianMixture model detected a few very low Net debt % EV values, and on the upper side as expected by the initial histogram, some extreme values.

Below code removes the list of extreme values above from our sample.


```python
pe_data_new = pe_data_new[densities >= threshold]
print('The sample size after outlier removal is', pe_data_new.shape[0],'. In total', pe_data.shape[0] - pe_data_new.shape[0], 'outliers were removed.')
```

    The sample size after outlier removal is 328 . In total 37 outliers were removed.
    

### 2.2.3 Histograms after outlier removal
Below code shows the histograms after outlier removal


```python
print('The sample size after outlier removal is', pe_data_new.shape[0],'. In total', pe_data.shape[0] - pe_data_new.shape[0], 'outliers were removed.')
```

    The sample size after outlier removal is 328 . In total 37 outliers were removed.
    


```python
sns.histplot(pe_data_new[['Market_Cap_TTM_Net_Income']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_51_1.png)
    


- Extreme outliers in higher values for PE ratio removed, but still some positive skewness
    > Use log of PE ratios instead of PE ratio, to make data more Gaussian


```python
sns.histplot(np.log(pe_data_new[['Market_Cap_TTM_Net_Income']]))
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_53_1.png)
    


- The log of PE ratios looks more Gaussian, and we will use this for predictions


```python
sns.histplot(pe_data_new[['Net_Debt_perc_EV']])
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_55_1.png)
    


- Extreme outliers in higher values for Net debt as % of EV removed and data shows no extreme skewness anymore

### 2.3 Split into training and testing set

Below code splits our samples into training and testing sets.
We will use a 80% training and 20% testing split.

Note that we also transform our target PE ratios by taking the log.


```python
from sklearn.model_selection import train_test_split

X = pe_data_new[['Sector', 'Net_Income_Forecast_CAGR_10y', 'Avg_Net_Income_Margin_Forecast_10y', 'Beta_5y', 'Net_Debt_perc_EV']]
Y = np.log(pe_data_new[['Market_Cap_TTM_Net_Income']])
Y = Y.rename(columns={'Market_Cap_TTM_Net_Income': 'Log_Market_Cap_TTM_Net_Income'})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size = 0.3, random_state=5)
```


```python
X_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sector</th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>Industrials</td>
      <td>0.052889</td>
      <td>0.212410</td>
      <td>1.278543</td>
      <td>0.072059</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>Communication Services</td>
      <td>0.001419</td>
      <td>0.073721</td>
      <td>0.000000</td>
      <td>0.188264</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>Consumer Discretionary</td>
      <td>0.034982</td>
      <td>0.074732</td>
      <td>1.206839</td>
      <td>0.181728</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>Materials</td>
      <td>0.106755</td>
      <td>0.150060</td>
      <td>1.137892</td>
      <td>0.158060</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>Industrials</td>
      <td>0.110710</td>
      <td>0.288177</td>
      <td>1.059018</td>
      <td>0.081942</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log_Market_Cap_TTM_Net_Income</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>3.320017</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>2.797414</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>3.698240</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>3.461179</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>3.326395</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data = Y_train.merge(X_train, how='outer', left_index=True, right_index=True)
test_data = Y_test.merge(X_test, how='outer', left_index=True, right_index=True)
```

## 3. Discover and visualize the data to gain insights

### 3.1 Summary Statistics


```python
train_data.describe().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log_Market_Cap_TTM_Net_Income</th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>229.00</td>
      <td>229.00</td>
      <td>229.00</td>
      <td>229.00</td>
      <td>229.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.39</td>
      <td>0.08</td>
      <td>0.18</td>
      <td>0.94</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.59</td>
      <td>0.08</td>
      <td>0.10</td>
      <td>0.44</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.19</td>
      <td>-0.20</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.15</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.01</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>0.65</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.32</td>
      <td>0.07</td>
      <td>0.16</td>
      <td>0.96</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.70</td>
      <td>0.11</td>
      <td>0.25</td>
      <td>1.22</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.03</td>
      <td>0.67</td>
      <td>0.57</td>
      <td>2.31</td>
      <td>0.48</td>
    </tr>
  </tbody>
</table>
</div>



- median log PE ratio is 3.32, ranging from 2.19 to 5.03
- median expected annual growth over the next 10 years is 7%, ranging from -20% to +67%
- median expected Net income margin over the next 10 years is 16%, ranging from -1% to +57%
- median Beta is 0.96, ranging from 0 to 2.31
- median Net Debt as a % of EV is 13%, ranging from -15% to +48%

Below we will analyse the training data, and analyse how the individual variables relate to our target PE ratio variable.

### 3.2 Correlation Matrix


```python
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Compute the correlation matrix
corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <AxesSubplot:>




    
![png](output_69_1.png)
    


Looking at the first column of the correlation matrix, we can analyse how individual variables are correlated with our target value of the Log of PE ratios
- The growth variable 'Net_Income_Forecast_CAGR_10y' is positively correlated with the PE ratio, as we would expect from financial theory
- The expected margin variable, our proxy for efficient investment and operations is also positively correlated with the PE ratio, as we would expect from financial theory
- Surprisingly the correlation between the Beta in the past 5 years and the PE ratio is very weak and close to 0. Based on financial theory we would have expected a negative correlation, as a higher beta would indicate higher risk
- The variable measuring the amount of debt a company has, 'Net_Debt_perc_EV', is negatively correlated with the PE ratio as we would expect based on financial theory, as companies with higher debt have a higher risk of bankruptcy and also higher expected debt repayments in the future, which reduces future cash flows for investors



```python
train_data.corr().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log_Market_Cap_TTM_Net_Income</th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Log_Market_Cap_TTM_Net_Income</th>
      <td>1.00</td>
      <td>0.43</td>
      <td>0.27</td>
      <td>0.08</td>
      <td>-0.36</td>
    </tr>
    <tr>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <td>0.43</td>
      <td>1.00</td>
      <td>0.28</td>
      <td>0.14</td>
      <td>-0.29</td>
    </tr>
    <tr>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <td>0.27</td>
      <td>0.28</td>
      <td>1.00</td>
      <td>-0.06</td>
      <td>-0.41</td>
    </tr>
    <tr>
      <th>Beta_5y</th>
      <td>0.08</td>
      <td>0.14</td>
      <td>-0.06</td>
      <td>1.00</td>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>Net_Debt_perc_EV</th>
      <td>-0.36</td>
      <td>-0.29</td>
      <td>-0.41</td>
      <td>-0.12</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



### 3.3 Pairplots


```python
sns.set_theme()
sns.jointplot(data=train_data, x="Net_Income_Forecast_CAGR_10y", y="Log_Market_Cap_TTM_Net_Income")
```




    <seaborn.axisgrid.JointGrid at 0x1f130a78e08>




    
![png](output_73_1.png)
    


- Scatterplot indicates positive relationship between expected growth and PE ratios, as indicated by the correlation matrix


```python
sns.jointplot(data=train_data, x="Avg_Net_Income_Margin_Forecast_10y", y="Log_Market_Cap_TTM_Net_Income")
```




    <seaborn.axisgrid.JointGrid at 0x1f12eb37148>




    
![png](output_75_1.png)
    


- Scatterplot indicates positive relationship between expected Net Income margins and PE ratios, as indicated by the correlation matrix


```python
sns.jointplot(data=train_data, x="Beta_5y", y="Log_Market_Cap_TTM_Net_Income")
```




    <seaborn.axisgrid.JointGrid at 0x1f130f72dc8>




    
![png](output_77_1.png)
    


- Scatterplot does not indicate any strong relationship either way between past 5 year Beta values and PE ratios, as indicated by the correlation matrix


```python
sns.jointplot(data=train_data, x="Net_Debt_perc_EV", y="Log_Market_Cap_TTM_Net_Income")
```




    <seaborn.axisgrid.JointGrid at 0x1f1310cf608>




    
![png](output_79_1.png)
    


- Scatterplot indicates negative relationship between Net debt as % of EV and PE ratios, as indicated by the correlation matrix

## 4. Prepare the data for Machine Learning algorithms

Before training our Machine learning algorithms we will do two pre-processing steps.

1. The categorical variable "Sector" will be transformed into a numerical variable with the OneHotEncoder, indicating the sector that applies to each company with 1, and the sectors that don't apply with 0
2. The numerical explanatory variables will be scaled with the StandardScaler to ensure that all new features have mean 0 and standard deviation 1. The scaler is trained on the training set and used to transform both the training and test set

The data will also be split into X and Y variables, Y being the PE ratio which is to be predicted, and X being the features used to predict the PE ratio.


```python
from sklearn.preprocessing import OneHotEncoder

train_data_cat = train_data[['Sector']]
test_data_cat = test_data[['Sector']]

#fit one hot encoder to training set
one_hot = OneHotEncoder(handle_unknown='ignore').fit(train_data_cat)

#transform training set
train_data_cat_one_hot = pd.DataFrame(one_hot.transform(train_data_cat).toarray())
train_data_cat_one_hot.columns = one_hot.get_feature_names(train_data_cat.columns)
train_data_cat_one_hot['Name'] = train_data_cat.index
train_data_cat_one_hot = train_data_cat_one_hot.set_index('Name')

#transform testing set
test_data_cat_one_hot = pd.DataFrame(one_hot.transform(test_data_cat).toarray())
test_data_cat_one_hot.columns = one_hot.get_feature_names(test_data_cat.columns)
test_data_cat_one_hot['Name'] = test_data_cat.index
test_data_cat_one_hot = test_data_cat_one_hot.set_index('Name')
```


```python
train_data_cat_one_hot.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sector_Communication Services</th>
      <th>Sector_Consumer Discretionary</th>
      <th>Sector_Consumer Staples</th>
      <th>Sector_Energy</th>
      <th>Sector_Healthcare</th>
      <th>Sector_Industrials</th>
      <th>Sector_Information Technology</th>
      <th>Sector_Materials</th>
      <th>Sector_Real Estate</th>
      <th>Sector_Utilities</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data_cat_one_hot.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sector_Communication Services</th>
      <th>Sector_Consumer Discretionary</th>
      <th>Sector_Consumer Staples</th>
      <th>Sector_Energy</th>
      <th>Sector_Healthcare</th>
      <th>Sector_Industrials</th>
      <th>Sector_Information Technology</th>
      <th>Sector_Materials</th>
      <th>Sector_Real Estate</th>
      <th>Sector_Utilities</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Eastman Chemical Company</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Allegion plc</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hormel Foods Corporation</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Extra Space Storage Inc.</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Target Corporation</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler

train_data_num = train_data[['Net_Income_Forecast_CAGR_10y','Avg_Net_Income_Margin_Forecast_10y','Beta_5y', 'Net_Debt_perc_EV']]
test_data_num = test_data[['Net_Income_Forecast_CAGR_10y','Avg_Net_Income_Margin_Forecast_10y','Beta_5y', 'Net_Debt_perc_EV']]

#fit standard scaler to training set
scaler = StandardScaler().fit(train_data_num)

#transform training set
train_data_num_scaled = pd.DataFrame(scaler.transform(train_data_num), columns = train_data_num.columns)
train_data_num_scaled['Name'] = train_data.index
train_data_num_scaled = train_data_num_scaled.set_index('Name')

#transform testing set
test_data_num_scaled = pd.DataFrame(scaler.transform(test_data_num), columns = test_data_num.columns)
test_data_num_scaled['Name'] = test_data.index
test_data_num_scaled = test_data_num_scaled.set_index('Name')
```


```python
train_data_num_scaled.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>-0.343975</td>
      <td>0.322551</td>
      <td>0.769134</td>
      <td>-0.517736</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>-0.967412</td>
      <td>-1.005990</td>
      <td>-2.133483</td>
      <td>0.282818</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>-0.560871</td>
      <td>-0.996304</td>
      <td>0.606348</td>
      <td>0.237788</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>0.308487</td>
      <td>-0.274720</td>
      <td>0.449821</td>
      <td>0.074738</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>0.356389</td>
      <td>1.048336</td>
      <td>0.270757</td>
      <td>-0.449652</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data_num_scaled.describe().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>229.00</td>
      <td>229.00</td>
      <td>229.00</td>
      <td>229.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-0.00</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.38</td>
      <td>-1.85</td>
      <td>-2.13</td>
      <td>-2.03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.56</td>
      <td>-0.76</td>
      <td>-0.66</td>
      <td>-0.73</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.13</td>
      <td>-0.18</td>
      <td>0.05</td>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.32</td>
      <td>0.65</td>
      <td>0.63</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.10</td>
      <td>3.75</td>
      <td>3.11</td>
      <td>2.32</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = train_data_num_scaled.merge(train_data_cat_one_hot, how='outer', left_index=True, right_index=True)
X_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
      <th>Sector_Communication Services</th>
      <th>Sector_Consumer Discretionary</th>
      <th>Sector_Consumer Staples</th>
      <th>Sector_Energy</th>
      <th>Sector_Healthcare</th>
      <th>Sector_Industrials</th>
      <th>Sector_Information Technology</th>
      <th>Sector_Materials</th>
      <th>Sector_Real Estate</th>
      <th>Sector_Utilities</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>-0.343975</td>
      <td>0.322551</td>
      <td>0.769134</td>
      <td>-0.517736</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>-0.967412</td>
      <td>-1.005990</td>
      <td>-2.133483</td>
      <td>0.282818</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>-0.560871</td>
      <td>-0.996304</td>
      <td>0.606348</td>
      <td>0.237788</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>0.308487</td>
      <td>-0.274720</td>
      <td>0.449821</td>
      <td>0.074738</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>0.356389</td>
      <td>1.048336</td>
      <td>0.270757</td>
      <td>-0.449652</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y_train = train_data[['Log_Market_Cap_TTM_Net_Income']]
Y_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log_Market_Cap_TTM_Net_Income</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMETEK, Inc.</th>
      <td>3.320017</td>
    </tr>
    <tr>
      <th>Fox Corporation</th>
      <td>2.797414</td>
    </tr>
    <tr>
      <th>Ralph Lauren Corporation</th>
      <td>3.698240</td>
    </tr>
    <tr>
      <th>The Sherwin-Williams Company</th>
      <td>3.461179</td>
    </tr>
    <tr>
      <th>Roper Technologies, Inc.</th>
      <td>3.326395</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test = test_data_num_scaled.merge(test_data_cat_one_hot, how='outer', left_index=True, right_index=True)
X_test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Net_Income_Forecast_CAGR_10y</th>
      <th>Avg_Net_Income_Margin_Forecast_10y</th>
      <th>Beta_5y</th>
      <th>Net_Debt_perc_EV</th>
      <th>Sector_Communication Services</th>
      <th>Sector_Consumer Discretionary</th>
      <th>Sector_Consumer Staples</th>
      <th>Sector_Energy</th>
      <th>Sector_Healthcare</th>
      <th>Sector_Industrials</th>
      <th>Sector_Information Technology</th>
      <th>Sector_Materials</th>
      <th>Sector_Real Estate</th>
      <th>Sector_Utilities</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Eastman Chemical Company</th>
      <td>2.163536</td>
      <td>-0.066516</td>
      <td>1.378961</td>
      <td>1.212203</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Allegion plc</th>
      <td>-0.299303</td>
      <td>-0.073863</td>
      <td>0.428535</td>
      <td>-0.234235</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hormel Foods Corporation</th>
      <td>-0.557665</td>
      <td>-0.728099</td>
      <td>-2.168545</td>
      <td>-1.114877</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Extra Space Storage Inc.</th>
      <td>-0.840145</td>
      <td>1.915674</td>
      <td>-1.576251</td>
      <td>0.767138</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Target Corporation</th>
      <td>-0.384196</td>
      <td>-1.276158</td>
      <td>-0.202580</td>
      <td>-0.285802</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y_test = test_data[['Log_Market_Cap_TTM_Net_Income']]
Y_test.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log_Market_Cap_TTM_Net_Income</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Eastman Chemical Company</th>
      <td>3.005154</td>
    </tr>
    <tr>
      <th>Allegion plc</th>
      <td>3.491612</td>
    </tr>
    <tr>
      <th>Hormel Foods Corporation</th>
      <td>3.355020</td>
    </tr>
    <tr>
      <th>Extra Space Storage Inc.</th>
      <td>3.559280</td>
    </tr>
    <tr>
      <th>Target Corporation</th>
      <td>3.121729</td>
    </tr>
  </tbody>
</table>
</div>



### 5. Select, train and fine-tune models
Now it's finally time to train our machine learning models.
As this is a regression task, we will take into considering below models.
1. Ridge Regression (Regularize linear regression)
2. KNeighborsRegressor
3. SVR linear
4. SVR rbf
5. RandomForestRegressor
6. Ensemble of models 1-5

- To optimize the hyper parameters, we will use grid search with fivefold cross validation. 
- The metric we will use to evaluate/score the performance is the r-squared of the model. The r-squared is the proportion of the variance of the target variable log PE ratio explained by our model.


```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge(max_iter = 100000)
grid_values = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 400, 500]}

# default metric to optimize over grid parameters: accuracy
grid_ridge = GridSearchCV(ridge, param_grid = grid_values, cv=5, scoring = 'r2')
grid_ridge.fit(X_train, Y_train.values.ravel());

print('Grid best parameter: ', grid_ridge.best_params_)
print('Grid best score: ', grid_ridge.best_score_.round(3))
```

    Grid best parameter:  {'alpha': 10}
    Grid best score:  0.221
    


```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
grid_values = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]}

# default metric to optimize over grid parameters: accuracy
grid_knn = GridSearchCV(knn, param_grid = grid_values, cv=5, scoring = 'r2')
grid_knn.fit(X_train, Y_train.values.ravel());

print('Grid best parameter: ', grid_knn.best_params_)
print('Grid best score: ', grid_knn.best_score_.round(3))
```

    Grid best parameter:  {'n_neighbors': 7}
    Grid best score:  0.234
    


```python
from sklearn.svm import SVR

svr1 = SVR(kernel='linear')
grid_values = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 400, 500]}

# default metric to optimize over grid parameters: accuracy
grid_svr1 = GridSearchCV(svr1, param_grid = grid_values, cv=5, scoring = 'r2')
grid_svr1.fit(X_train, Y_train.values.ravel());

print('Grid best parameter: ', grid_svr1.best_params_)
print('Grid best score: ', grid_svr1.best_score_.round(3))
```

    Grid best parameter:  {'C': 400}
    Grid best score:  0.202
    


```python
from sklearn.svm import SVR

svr2 = SVR(kernel='rbf')
grid_values = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 400, 500]}

# default metric to optimize over grid parameters: accuracy
grid_svr2 = GridSearchCV(svr2, param_grid = grid_values, cv=5, scoring = 'r2')
grid_svr2.fit(X_train, Y_train.values.ravel());

print('Grid best parameter: ', grid_svr2.best_params_)
print('Grid best score: ', grid_svr2.best_score_.round(3))
```

    Grid best parameter:  {'C': 1}
    Grid best score:  0.246
    


```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
grid_values = {'max_depth': [2, 3, 4, 5, 6, 7, 8], 
               'n_estimators': [50, 75, 100, 125, 150]}

# default metric to optimize over grid parameters: accuracy
grid_rf = GridSearchCV(rf, param_grid = grid_values, cv=5, scoring = 'r2')
grid_rf.fit(X_train, Y_train.values.ravel())

print('Grid best parameter: ', grid_rf.best_params_)
print('Grid best score: ', grid_rf.best_score_.round(3))
```

    Grid best parameter:  {'max_depth': 8, 'n_estimators': 100}
    Grid best score:  0.27
    


```python
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score

er = VotingRegressor([('lr_ridge', grid_ridge), ('knn', grid_knn), ('svr_linear', grid_svr1), 
                                     ('svr_rbf', grid_svr2), ('rf', grid_rf)])
er.fit(X_train, Y_train.values.ravel())
cv_score = cross_val_score(er, X_train, Y_train.values.ravel(), cv=3, scoring = 'r2').mean().round(3)
print('CV score', cv_score)
```

    CV score 0.284
    

### 6. Conclusion

Based on the average cross validation r-squared from five-fold cross validation, we get the below r-squared for each model.


```python
classifiers = ['Ridge', 
               'KNeighborsRegressor',
               'SVR linear',
               'SVR rbf', 
               'RandomForestClassifier', 
               'VotingClassifier']

scores = [grid_ridge.best_score_.round(3), 
          grid_knn.best_score_.round(3), 
          grid_svr1.best_score_.round(3), 
          grid_svr2.best_score_.round(3), 
          grid_rf.best_score_.round(3), 
          cv_score.round(3)]

model_scores = pd.DataFrame(data= scores, columns = ['CV_Accuracy'], index = classifiers)
model_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CV_Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ridge</th>
      <td>0.221</td>
    </tr>
    <tr>
      <th>KNeighborsRegressor</th>
      <td>0.234</td>
    </tr>
    <tr>
      <th>SVR linear</th>
      <td>0.202</td>
    </tr>
    <tr>
      <th>SVR rbf</th>
      <td>0.246</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.270</td>
    </tr>
    <tr>
      <th>VotingClassifier</th>
      <td>0.284</td>
    </tr>
  </tbody>
</table>
</div>



The model with the highest cross validation r-squared score is the ensemble model with an r-squared of 28.4%. We will use this model to predict the out of sample error on the test data.


```python
er.score(X_test, Y_test).round(3)
```




    0.221



Both our cross validation r-squared values in predicting the log of the PE ratios, as well as the out of sample r-squared estimation on the test set indicate that our fundamental financial explanatory variables relating to expected growth, expected margins, beta and the level of debt a company has explains only 22% of the variance in the PE ratios.

This illustrates that financial data is noisy, and 78% of the variance in the PE ratios remains unexplained with our variables.


```python

```
