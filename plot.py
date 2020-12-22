#
import matplotlib.pyplot as plt
import csv
import numpy as np

import pandas as pd
import pandas_datareader.data as web
import datetime
import quandl as q

q.ApiConfig.api_key = "8daXuHDtVszfAfF_KanM"
# send a get request to query Microsoft's end of day stock prices from 1st
# Jan, 2010 to 1st Jan, 2019
msft_data = q.get("EOD/MSFT", start_date="2010-01-01", end_date="2019-01-01")
btc_data = q.get("BITSTAMP/USD", start_date="2020-11-15", end_date="2020-12-21")

# look at the first 5 rows of the dataframe
msft_data.head()
btc_data.head()

msft_data.describe()
btc_data.describe()

msft_data.resample('M').mean()
btc_data.resample('M').mean()

# assign `Adj Close` to `daily_close`
# daily_close = msft_data[['Adj_Close']]
daily_close = btc_data[['Last']]
delta = daily_close.diff()

# To find +ve & -ve gains
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0

# Calc EWMA
window_length = 14
roll_up1 = up.ewm(span=window_length).mean()
roll_down1 = down.abs().ewm(span=window_length).mean()

# Calc RSI based on EWMA
RS1 = roll_up1/roll_down1
RSI1 = 100 - (100 / (1 + RS1))

# Calc SMA
roll_up2 = up.rolling(window_length).mean()
roll_down2 = down.abs().rolling(window_length).mean()

# Calc RSI based on SMA
RS2 = roll_up2/roll_down2
RSI2 = 100 - (100 / (1 + RS2))

# -----------------------------------------------------------------------------
# returns as fractional change
daily_return = daily_close.pct_change()

# replacing NA values with 0
daily_return.fillna(0, inplace=True)
# print(daily_return)

# mdata = msft_data.resample('M').apply(lambda x: x[-1])
mdata = btc_data.resample('M').apply(lambda x: x[-1])

monthly_return = mdata.pct_change()

# assigning adjusted closing prices to adj_prices
# adj_price = msft_data['Adj_Close']
adj_price = btc_data['Last']
# print(adj_price)
# calculate the moving average

# the below code only selects numeric data, this method should not be used
# often in case there are other columns with numeric data, in which case
# use input[:,col_of_choice] after the below loc (line of code)
input = np.array(adj_price)

# input = input[:,1]
mav = adj_price.rolling(window=50).mean()

# print the result
# print(mav[-10:])
# adj_price.plot()

# mav.plot()

# --------------

maria = []
davidic = []
his = []
chelsea = []

data = np.genfromtxt('D:/King/im_values.csv', delimiter=',')

print(data[2][0])

for i in range(len(data)-2):
    davidic.append(data[i+2][0])

# ----------------------------------
# Modified RSI Strategy Algorithm

# a = davidic
a = input
a_pg = 1
a_pl = 1
prev_avg_gain = 0
prev_avg_loss = 0

RSI_1 = []

# Keeping days one and two averaged over the entire dataset, before beginning prediction of trend
RSI_1.append((sum(a)/len(a))/max(a)*100)
RSI_1.append((sum(a)/len(a))/max(a)*100)

RSI_2 = RSI_1
RSI_3 = RSI_1

count1 = 0
count2 = 0
for i in range(len(a)):

    # Original RSI attempt by moi (gave errors)
    if i > 1:
        if a[i - 1] > a[i - 2]:  # to calculate percentage gain over a two day period
            a_pg = (a[i - 1] + a[i - 2]) / 2 * 100
            count1 = count1 + a_pg

        else:  # % l
            a_pl = (a[i - 1] + a[i - 2]) / 2 * 100
            count2 = count2 + a_pl

    for j in range(i-3):
        # from HMM, a trend is solidified after 2 consecutive days of fixed price: so to calculate
        # avg gain periodically we'll look at the past 2 days and its average will be compared with
        # current day for DAY ONE avg gain/l
        # avg_prev2 = ((a[j-1])+(a[j-2]))/2
        curr = a[j]

        # # Expressing the below formula:
        # if curr>avg_prev2 && curr>( a(j-1)*0.9 + a(j-2)*0.1 ) # ensures that the current price is
        # actually being compared to the previous day's price which has been given higher weight,
        # 0.9 over the previous previous day's price which has been given a lesser weight.

        # Modify this formula so it changes iteratively, for example if n days are available then:
        # a1 + a2x1 + a3x^2 + a4x^3.. where coefficients will be adjusted weights,
        # 0.5*yesterday + 0.3*daybeforeyesterday + 0.2*second_dby + 0.1*third_dby

        # How to calculate this formula w/out making 14 nested if-else statements: polyfit, but this
        # creates a non-exponential? say if we want yesterday's price-weight to be half of today's
        # weight, or rather a very large fraction of the overall weight. Or lemme just use 5 days,
        # cause after that then the 7th day will be too small a weight to cause a dent.

        # Modified RSI
        # if j == 2:
        #     a_t = 0.9*a[j-1] + 0.1*a[j-2]  # a_t = trend
        #
        #     if a[j-1] > a[j-2]:  # to calculate percentage gain over a two day period
        #         a_pg = (a[j-1]+a[j-2])/2 * 100
        #
        #     else:  # % l
        #         a_pl = (a[j-1]+a[j-2])/2 * 100
        #
        # elif j == 3:
        #     a_t = 0.7*a[j-1] + 0.2*a[j-2] + 0.1*a[j-3]
        #
        #     if a[j-1] > (0.7*a[j-2] + 0.3*a[j-3]):
        #         a_pg = (a[j-1]+a[j-2]+a[j-3])/3 * 100
        #
        #     else: # % l
        #         a_pl = (a[j-1]+a[j-2]+a[j-3])/3 * 100
        #
        # elif j == 4:
        #     a_t = 0.55*a[j-1] + 0.25*a[j-2] + 0.15*a[j-3] + 0.5*a[j-4]
        #
        #     if a[j-1] > (0.6*a[j-2] + 0.3*a[j-3] + 0.1*a[j-4]):
        #         a_pg = (a[j-1]+a[j-2]+a[j-3]+a[j-4])/4 * 100
        #
        #     else: # % l
        #         a_pl = (a[j-1]+a[j-2]+a[j-3]+a[j-4])/4 * 100
        #
        # elif j > 4:
        #     a_t = 0.5*a[j-1] + 0.2*a[j-2] + 0.15*a[j-3] + 0.1*a[j-4] + 0.05*a[j-5]
        #
        #     if a[j-1] > (0.55*a[j-2] + 0.2*a[j-3] + 0.1*a[j-4]) + 0.05*a[j-5]:
        #         a_pg = (a[j-1]+a[j-2]+a[j-3]+a[j-4]+a[j-5])/5 * 100
        #
        #     else: # % l
        #         a_pl = (a[j-1]+a[j-2]+a[j-3]+a[j-4]+a[j-5])/5 * 100
        #
        # if curr>avg_prev2 && curr> a_t
        # count = [count

    # what_number_to_put_here = (i if i<6 OR 5) i.e
    what_number_to_put_here = i*(i < 6) + 5*(i >= 6)

    if i >= 2:
        RSI_1.append( 100 - (100/(1+(a_pg/what_number_to_put_here)/(a_pl/what_number_to_put_here) )) )

        RSI_2.append(100 - (100/(1+ ( (prev_avg_gain * 13) + a_pg)/( (prev_avg_loss * 13) + a_pl) )))

        RSI_3.append( 100 - (100/(1+(count1/i)/(count2/i) )) )
        prev_avg_gain = a_pg
        prev_avg_loss = a_pl

# ----------------------------------

fig, axs = plt.subplots(3)

for i in range(len(a)):
    maria.append(i+1)

for i in range(len(RSI_3)):
    his.append(i+1)

for i in range(len(RSI1)):
    chelsea.append(i+1)

axs[0].set_title('BTC/USD prices from 15 Nov to Aunty Lolia\'s birthday')
axs[0].plot(maria, a)

axs[1].set_title('modified RSI_3')
axs[1].plot(his, RSI_3)

axs[2].set_title('RSI_true')
axs[2].plot(chelsea, RSI1)
plt.show()



