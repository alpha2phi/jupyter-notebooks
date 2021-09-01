
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy import optimize, stats
from sklearn import preprocessing, svm
import statsmodels.api as sm
import statsmodels.tsa.api as smt

pd.plotting.register_matplotlib_converters()


###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
'''
Plot ts with rolling mean and 95% confidence interval with rolling std.
:parameter
    :param ts: pandas Series
    :param window: num for rolling stats
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(15,5)):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(ts.name)
    plt.plot(ts[window:], label='Actual values', color="black")
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window), color="red")
    if plot_intervals:
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound, color='lightskyblue', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


'''
Test stationarity by:
    - running Augmented Dickey-Fuller test wiht 95%
    - plotting mean and variance of a sample from data
    - plottig autocorrelation and partial autocorrelation
'''
def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(15,10)):
    with plt.style.context(style='bmh'):
        ## set figure
        fig = plt.figure(figsize=figsize)
        ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
        acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))

        ## plot ts with mean/std of a sample from the first x%
        dtf_ts = ts.to_frame(name="ts")
        sample_size = int(len(ts)*sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red", linestyle="--", linewidth=0.7)
        ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(ax=ts_ax, legend=False, color="red", linewidth=0.9)
        ts_ax.fill_between(x=dtf_ts.head(sample_size).index, y1=dtf_ts['lower'].head(sample_size), y2=dtf_ts['upper'].head(sample_size), color='lightskyblue')

        ## test stationarity (Augmented Dickey-Fuller)
        adfuller_test = sm.tsa.stattools.adfuller(ts, maxlag=maxlag, autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+' (p-value: '+str(p)+')')

        ## pacf (for AR) e acf (for MA)
        smt.graphics.plot_pacf(ts, lags=maxlag, ax=pacf_ax, title="Partial Autocorrelation (for AR component)")
        smt.graphics.plot_acf(ts, lags=maxlag, ax=acf_ax, title="Autocorrelation (for MA component)")
        plt.tight_layout()


'''
Defferenciate ts.
:parameter
    :param ts: pandas Series
    :param lag: num - diff[t] = y[t] - y[t-lag]
    :param order: num - how many times it has to differenciate: diff[t]^order = diff[t] - diff[t-lag]
    :param drop_na: logic - if True Na are dropped, else are filled with last observation
'''
def diff_ts(ts, lag=1, order=1, drop_na=True):
    for i in range(order):
        ts = ts - ts.shift(lag)
    ts = ts[(pd.notnull(ts))] if drop_na is True else ts.fillna(method="bfill")
    return ts


'''
'''
def undo_diff(ts, first_y, lag=1, order=1):
    for i in range(order):
        (24168.04468 - 18256.02366) + a.cumsum()
        ts = np.r_[ts, ts[lag:]].cumsum()
    return ts

'''
Run Granger test on 2 series
'''
def test_2ts_casuality(ts1, ts2, maxlag=30, figsize=(15,5)):
    ## prepare
    dtf = ts1.to_frame(name=ts1.name)
    dtf[ts2.name] = ts2
    dtf.plot(figsize=figsize, grid=True, title=ts1.name+"  vs  "+ts2.name)
    plt.show()
    ## test casuality (Granger test)
    granger_test = sm.tsa.stattools.grangercausalitytests(dtf, maxlag=maxlag, verbose=False)
    for lag,tupla in granger_test.items():
        p = np.mean([tupla[0][k][1] for k in tupla[0].keys()])
        p = round(p, 3)
        if p < 0.05:
            conclusion = "Casuality with lag "+str(lag)+" (p-value: "+str(p)+")"
            print(conclusion)

'''
Decompose ts into
    - trend component = moving avarage
    - seasonality
    - residuals = y - (trend + seasonality)
:parameter
    :param s: num - number of observations per season (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
'''
def decompose_ts(ts, s=250, figsize=(20,13)):
    decomposition = smt.seasonal_decompose(ts, freq=s)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(ts)
    ax[0].set_title('Original')
    ax[0].grid(True)
    ax[1].plot(trend)
    ax[1].set_title('Trend')
    ax[1].grid(True)
    ax[2].plot(seasonal)
    ax[2].set_title('Seasonality')
    ax[2].grid(True)
    ax[3].plot(residual)
    ax[3].set_title('Residuals')
    ax[3].grid(True)
    return {"trend":trend, "seasonal":seasonal, "residual":residual}



'''
Find outliers using sklearn unsupervised support vetcor machine.
:parameter
    :param ts: pandas Series
    :param perc: float - percentage of outliers to look for
:return
    dtf with raw ts, outlier 1/0 (yes/no), numeric index
'''
def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda x: 1 if x==-1 else 0)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found "+str(sum(dtf_outliers["outlier"]==1)))
    ax.plot(dtf_outliers["index"], dtf_outliers["ts"], color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1]["index"], y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'], color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers



'''
Interpolate outliers in a ts.
'''
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5, title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black", label="interpolated", legend=True)
    plt.show()
    return ts_clean



###############################################################################
#                 MODEL DESIGN & TESTING - FORECASTING                        #
###############################################################################
'''
Split train/test from any given data point.
:parameter
    :param ts: pandas Series
    :param exog: array len(ts) x n regressors
    :param test: num or str - test size (ex. 0.20) or index position (ex. "yyyy-mm-dd", 1000)
:return
    ts_train, ts_test, exog_train, exog_test
'''
def split_train_test(ts, exog=None, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|", ts.index[split], "| test size:", perc, " ---")

    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=figsize)
        ts_train.plot(ax=ax[0], grid=True, title="Train", color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test", color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()

    ## split exog
    if exog is not None:
        exog_train = exog[0:split]
        exog_test = exog[split:]
        return ts_train, ts_test, exog_train, exog_test
    else:
        return ts_train, ts_test



###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet on Business Data:
    y = trend + seasonality + holidays
:parameter
    :param dtf_train: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param dtf_test: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param lst_exog: list - names of variables
    :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS" monthly start ...
:return
    dtf with predictons and the model
'''
def fit_prophet(dtf_train, dtf_test, lst_exog=None, model=None, freq="D", conf=0.95, figsize=(15,10)):
    ## setup prophet
    if model is None:
        model = Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                        holidays=None, interval_width=conf)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)

    ## train
    model.fit(dtf_train)

    ## test
    dtf_prophet = model.make_future_dataframe(periods=len(dtf_test)+10, freq=freq, include_history=True)

    if model.growth == "logistic":
        dtf_prophet["cap"] = dtf_train["cap"].unique()[0]

    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(dtf_train[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-len(dtf_test):][lst_exog] = dtf_test[lst_exog].values

    dtf_prophet = model.predict(dtf_prophet)
    dtf_train = dtf_train.merge(dtf_prophet[["ds","yhat"]], how="left").rename(
        columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    dtf_test = dtf_test.merge(dtf_prophet[["ds","yhat","yhat_lower","yhat_upper"]], how="left").rename(
        columns={'yhat':'forecast', 'y':'ts', 'yhat_lower':'lower', 'yhat_upper':'upper'}).set_index("ds")

    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_ts_model(dtf, conf=conf, figsize=figsize, title="Prophet")
    return dtf, model



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_prophet(dtf, model, pred_ahead=None, end=None, freq="D", conf=0.95, zoom=30, figsize=(15,5)):
    ## fit
    model.fit(dtf)

    ## index
    index = utils_generate_indexdate(start=dtf["ds"].values[-1], end=end, n=pred_ahead, freq=freq)

    ## forecast
    dtf_prophet = model.make_future_dataframe(periods=len(index), freq=freq, include_history=True)
    dtf_prophet = model.predict(dtf_prophet)
    dtf = dtf.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    preds = pd.DataFrame(data=index, columns=["ds"])
    preds = preds.merge(dtf_prophet[["ds","yhat","yhat_lower","yhat_upper"]], how="left").rename(
        columns={'yhat':'forecast', 'yhat_lower':'lower', 'yhat_upper':'upper'}).set_index("ds")
    dtf = dtf.append(preds)

    ## plot
    dtf = utils_add_forecast_int(dtf, conf=conf, zoom=zoom)
    return dtf



###############################################################################
#                    PARAMETRIC CURVE FITTING                                 #
###############################################################################
'''
Fits a custom function.
:parameter
    :param X: array
    :param y: array
    :param f: function to fit (ex. logistic: f(X) = capacity / (1 + np.exp(-k*(X - midpoint)))
                                or gaussian: f(X) = a * np.exp(-0.5 * ((X-mu)/sigma)**2)   )
    :param kind: str - "logistic", "gaussian" or None
    :param p0: array or list of initial parameters (ex. for logistic p0=[np.max(ts), 1, 1])
:return
    optimal params
'''
def fit_curve(X, y, f=None, kind=None, p0=None):
    ## define f(x) if not specified
    if f is None:
        if kind == "logistic":
            f = lambda p,X: p[0] / (1 + np.exp(-p[1]*(X-p[2])))
        elif find == "gaussian":
            f = lambda p,X: p[0] * np.exp(-0.5 * ((X-p[1])/p[2])**2)

    ## find optimal parameters
    model, cov = optimize.curve_fit(f, X, y, maxfev=10000, p0=p0)
    return model



'''
Predict with optimal parameters.
'''
def utils_predict_curve(model, f, X):
    fitted = f(X, model[0], model[1], model[2])
    return fitted



'''
Plot parametric fitting.
'''
def utils_plot_parametric(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ## entire series
    dtf["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")
    dtf["model"].plot(ax=ax[0], color="green", label="model", legend=True)
    dtf["forecast"].plot(ax=ax[0], grid=True, color="red", label="forecast", legend=True)
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)

    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black",
                                  title="Zoom on the last "+str(zoom)+" observations")
    dtf.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
    dtf.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'],
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    plt.show()
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param f: function
    :param model: list of optim params
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_curve(ts, f, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    fitted = utils_predict_curve(model, f, X=np.arange(len(ts)))
    dtf = ts.to_frame(name="ts")
    dtf["model"] = fitted

    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)

    ## forecast
    preds = utils_predict_curve(model, f, X=np.arange(len(ts)+1, len(ts)+1+len(index)))
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))

    ## plot
    utils_plot_parametric(dtf, zoom=zoom)
    return dtf



###############################################################################
#                              CLUSTERING                                     #
###############################################################################