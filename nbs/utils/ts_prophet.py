
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy import optimize, stats


'''
Compute the confidence interval for predictions:
    [y[t+h] +- (c*σ*√h)]
:parameter
    :param lst_values: list or array
    :param error_std: σ (standard dev of residuals)
    :param conf: num - confidence level (90%, 95%, 99%)
:return
    array with 2 columns (upper and lower bounds)
'''
def utils_conf_int(lst_values, error_std, conf=0.95):
    lst_values = list(lst_values) if type(lst_values) != list else lst_values
    c = round( stats.norm.ppf(1-(1-conf)/2), 2)
    lst_ci = []
    for x in lst_values:
        lst_x = lst_values[:lst_values.index(x)+1]
        h = len(lst_x)
        ci = [x - (c*error_std*np.sqrt(h)), x + (c*error_std*np.sqrt(h))]
        lst_ci.append(ci)
    return np.array(lst_ci)


'''
Evaluation metrics for predictions.
:parameter
    :param dtf: DataFrame with columns "ts", "model", "forecast", and "lower"/"upper" (if available)
:return
    dtf with columns "ts", "model", "residuals", "lower", "forecast", "upper", "error"
'''
def utils_evaluate_ts_model(dtf, conf=0.95, title=None, plot=True, figsize=(20,13)):
    try:
        ## residuals from fitting
        ### add column
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        ### kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()

        ## forecasting error
        ### add column
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        ### kpi
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["error"].apply(lambda x: x**2).mean()  #mean squared error
        rmse = np.sqrt(mse)  #root mean squared error

        ## interval
        if "upper" not in dtf.columns:
            print("--- computing confidence interval ---")
            dtf["lower"], dtf["upper"] = [np.nan, np.nan]
            dtf.loc[dtf["forecast"].notnull(), ["lower","upper"]] = utils_conf_int(
                dtf[dtf["forecast"].notnull()]["forecast"], residuals_std, conf)

        ## plot
        if plot is True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['lower'], y2=dtf['upper'], color='b', alpha=0.2)
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))

        return dtf[["ts", "model", "residuals", "lower", "forecast", "upper", "error"]]

    except Exception as e:
        print("--- got error ---")
        print(e)

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
