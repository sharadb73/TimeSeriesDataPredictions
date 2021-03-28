from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6
import pandas as pd
import numpy as np
import os


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def readData(path):
    print(path)
    dfLog = pd.DataFrame(
        columns=['GroupID', 'Instance', 'Timestamp', 'MemoryAllocated', 'MemoryUsed', 'CPUAllocated',
                 'CPUUsed', 'NetworkBandwidthUtilization', 'StorageSpaceUtilization'])
    dfLogTemp = pd.DataFrame(columns=['raw1', 'raw2', 'GroupID', 'Instance'])
    with os.scandir(path) as gid_insid:

        for gid_insid_name in gid_insid:
            ginssubfolder = path + "\\" + gid_insid_name.name
            print(ginssubfolder)
            ginsname = gid_insid_name.name.split('_')
            group = int(ginsname[1])
            # print(group)
            insid = ginsname[2]
            # print(insid)
            with os.scandir(ginssubfolder) as ginssub:
                for logfile in ginssub:
                    if logfile.is_file():
                        # print(logfile.name)
                        logpath = ginssubfolder + '\\' + logfile.name

                        dflogdata = pd.read_csv(logpath, delimiter=':', header=None, names=["raw1", "raw2"])
                        dflogdata["GroupID"] = group
                        dflogdata["Instance"] = insid

                        dfLogTemp = dfLogTemp.append(dflogdata, ignore_index=True, sort=False)

    dfLog['GroupID'] = dfLogTemp['GroupID']
    dfLog['Instance'] = dfLogTemp['Instance']
    dfLog['Timestamp'] = pd.to_datetime(dfLogTemp['raw1'].str.replace("IST", ""), format="%c", utc=False)
    dfLog[[
        'MemoryAllocated', 'MemoryUsed', 'CPUAllocated', 'CPUUsed', 'NetworkBandwidthUtilization',
        'StorageSpaceUtilization']] = \
        dfLogTemp['raw2'].str.split(':', expand=True)
    dfLog[['MemoryAllocated', 'MemoryUsed', 'CPUAllocated']] = dfLog[
        ['MemoryAllocated', 'MemoryUsed', 'CPUAllocated']].astype(int)
    dfLog[['CPUUsed', 'NetworkBandwidthUtilization']] = dfLog[['CPUUsed', 'NetworkBandwidthUtilization']].astype(float)
    dfLog['StorageSpaceUtilization'] = dfLog['StorageSpaceUtilization'].str.replace("G", "").astype(int)

    dfLog.sort_values(by=['GroupID', 'Instance', 'Timestamp'], inplace=True, ascending=True)
    # dfLog.to_excel('data.xlsx', index=False)
    # print(dfLog)
    dfLogTest = pd.DataFrame(columns=[])
    dfLogTest['Timestamp'] = pd.to_datetime(dfLog['Timestamp'], infer_datetime_format=True)
    dfLogTest['MemoryUsed'] = dfLog['MemoryUsed']

    dfLogTest = dfLogTest.set_index('Timestamp')
    # dfLog.plot(x='Timestamp', y=['GroupID', 'Instance', 'MemoryAllocated','MemoryUsed', 'CPUAllocated', 'CPUUsed', 'NetworkBandwidthUtilization', 'StorageSpaceUtilization'])
    # dfLogTest.plot(x='Timestamp', y=['MemoryUsed'])
    plt.xlabel('Date')
    plt.ylabel('Memory Used')
    plt.plot(dfLogTest)
    plt.close()

    # Determine rolling statistics
    rolmean = dfLogTest.rolling(window=365).mean()
    rolstd = dfLogTest.rolling(window=365).std()
    # print(rolmean, rolstd)

    orig = plt.plot(dfLogTest, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.close()

    # Determine Dickey-Fuller test
    print("Result of Dickey-Fuller Test:")
    dftest = adfuller(dfLogTest['MemoryUsed'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    print(dfoutput)

    dfLogTest_logScale = np.log(dfLogTest)
    plt.plot(dfLogTest_logScale)
    plt.close()

    movingAverage = dfLogTest_logScale.rolling(window=365).mean()
    movingSTD = dfLogTest_logScale.rolling(window=365).std()
    plt.plot(dfLogTest_logScale)
    plt.plot(movingAverage, color='red')
    plt.close()

    datasetLogScaleMinusMovingAverage = dfLogTest_logScale - movingAverage
    print(datasetLogScaleMinusMovingAverage.head(12))

    datasetLogScaleMinusMovingAverage.dropna(inplace=True)
    print(datasetLogScaleMinusMovingAverage.head(10))

    exponentialDecayWeightedAverage = dfLogTest_logScale.ewm(halflife=365, min_periods=0, adjust=True).mean()
    plt.plot(dfLogTest_logScale)
    plt.plot(exponentialDecayWeightedAverage, color='red')
    plt.close()

    datasetLogScaleMinusMovingExponentialDecayAverage = dfLogTest_logScale - exponentialDecayWeightedAverage
    test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

    datasetLogDiffShifting = dfLogTest_logScale - dfLogTest_logScale.shift()
    # test_stationarity(datasetLogDiffShifting)

    datasetLogDiffShifting.dropna(inplace=True)
    # test_stationarity(datasetLogDiffShifting)

    # decomposition = seasonal_decompose(dfLogTest_logScale)
    # trend = decomposition.trend
    # seasonal = decomposition.seasonal
    # residual = decomposition.resid
    #
    # plt.subplot(411)
    # plt.plot(dfLogTest_logScale, lable='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, lable='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal, lable='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, lable='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    #
    # decomposeLogData = residual
    # decomposeLogData.dropna(inplace=True)
    # test_stationarity(decomposeLogData)

    lag_acf = acf(datasetLogDiffShifting, nlags=20)
    lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols') # ordinary lest square method

    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
    plt.title('Autocorrection Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
    plt.title('Partial Autocorrection Function')
    plt.tight_layout()
    plt.close()

    # # AR Model Auto regressive
    # model = ARIMA(dfLogTest_logScale, order=(1, 1, 1))
    # results_AR = model.fit(disp=1)
    # plt.plot(datasetLogDiffShifting)
    # plt.plot(results_AR.fittedvalues, color='red')
    # plt.title('RSS: %.4f'% sum(results_AR.fittedvalues-datasetLogDiffShifting['MemoryUsed'])**2)
    # print('Plotting AR Model')
    # plt.close()
    #
    # # MA Model
    # model = ARIMA(dfLogTest_logScale, order=(1, 1, 1))
    # results_MA = model.fit(disp=1)
    # plt.plot(datasetLogDiffShifting)
    # plt.plot(results_MA.fittedvalues, color='red')
    # plt.title('RSS: %.4f'% sum(results_MA.fittedvalues-datasetLogDiffShifting['MemoryUsed'])**2)
    # print('Plotting AR Model')
    # plt.close()

    # ARIMA Model
    model = ARIMA(dfLogTest_logScale, order=(1, 1, 1))
    results_ARIMA = model.fit(disp=1)
    plt.plot(datasetLogDiffShifting)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum(results_ARIMA.fittedvalues-datasetLogDiffShifting['MemoryUsed'])**2)
    print('Plotting AR Model')
    plt.close()

    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    print(predictions_ARIMA_diff.head())

    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print(predictions_ARIMA_diff_cumsum.head())

    predictions_ARIMA_log = pd.Series(dfLogTest_logScale['MemoryUsed'].ix[0], index=dfLogTest_logScale.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_values=0)
    predictions_ARIMA_log.head()

    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(dfLogTest)
    plt.plot(predictions_ARIMA)
    plt.close()

    results_ARIMA.plot_predict(1, 47411)        # 47411 total actual data records + steps(days)
    x=results_ARIMA.forecast(steps=365)

    print(x)


def test_stationarity(timeseries):
    # Determine rolling statistics
    movingAverage = timeseries.rolling(window=365).mean()
    movingSTD = timeseries.rolling(window=365).std()

    # Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.close()

    # Perform Dickey-Fuller test
    print("Result of Dickey-Fuller Test:")
    dftest = adfuller(timeseries['MemoryUsed'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    print(dfoutput)


if __name__ == '__main__':
    readData('E:\Learning_Personal\ESDS Assignment\group82_resource_utilization')
