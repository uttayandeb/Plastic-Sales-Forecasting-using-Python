
                 ######## FORECASTING PLASTIC SALES ###############




##### IMPORTING the required packages and LOADING the requird Dataset ########

import pandas as pd
import numpy as np
import xlrd
import xlwt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

plastic=pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Forecasting\\PlasticSales.csv")


month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plastic['Month'][0]
p=plastic['Month'][0]
p
p[0:3]

plastic['month']=0

for i in range(60):
    p=plastic['Month'][i]
    plastic['month'][i]=p[0:3]
    
month_dummies= pd.DataFrame(pd.get_dummies(plastic['month']))

Plastic_Sales=pd.concat((plastic,month_dummies),axis=1)



Plastic_Sales['t']=np.arange(1,61)

Plastic_Sales['t_square']=Plastic_Sales['t']*Plastic_Sales['t']



Plastic_Sales['log_Sales']=np.log(Plastic_Sales['Sales'])


########## SPLITTING THE DATA INTO   T E S T    AND   T R A I N ##################

train=Plastic_Sales.head(48)
test=Plastic_Sales.tail(12)

Plastic_Sales.Sales.plot()


########################################################################
################### M O D E L     B U I L D I N G ######################
########################################################################






##################### LINEAR MODEL #########################

linear_model= smf.ols('Sales~t',data=train).fit()
linear_model
predlinear= pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_lin= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predlinear))**2))
rmse_lin ##260.9378142511123

#################### QUADRATIC MODEL #####################

quad_model=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad_model.predict(pd.DataFrame(test[['t','t_square']])))
predquad

rmse_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmse_quad #297.40670972721097

######################### EXPONENTIAL MODEL #################

exp_model=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(exp_model.predict(pd.DataFrame(test['t'])))
rmse_exp= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmse_exp #268.693838500261

######################## ADDITIVE SEASONALITY ##################

add_sea=smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
pred_addsea=pd.Series(add_sea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_add= np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_addsea))**2))
rmse_add #235.60267356646528


###################### ADDITIVE SEASONALITY WITH LINEAR TREND ###########

add_sealin=smf.ols('Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddlin=pd.Series(add_sealin.predict(pd.DataFrame(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmseaddlin= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlin))**2))
rmseaddlin #135.5535958348212



################## ADDITIVE SEASONALITY WITH QUADRATIC TREND ###############


add_seaquad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddquad= pd.Series(add_seaquad.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmseaddquad= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad #218.19387584891263

################### MULTIPLICATIVE SEASONALITY ########################

mul_lin= smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmul=pd.Series(mul_lin.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemul=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul  #239.65432143120915

################## MULTIPLICATIVE SEASONALITY WITH LINEAR TREND ############

mul_add= smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmuladd= pd.Series(mul_add.predict(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemuladd = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmuladd)))**2))
rmsemuladd #160.68332947192565

################# MULTIPLICATIVE SEASONALITY WITH QUADRATIC TREND ###########

mul_quad = smf.ols('log_Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmulquad= pd.Series(mul_add.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemulquad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmulquad)))**2))
rmsemulquad #160.68332947192565




#TABULAR FORM OF RMSE
data={'MODEL': pd.Series(['rmse_add','rmse_exp','rmse_lin','rmse_quad','rmseaddlin','rmseaddquad','rmsemul','rmsemuladd','rmsemulquad']), 'ERROR_VALUES':pd.Series([rmse_add,rmse_exp,rmse_lin,rmse_quad,rmseaddlin,rmseaddquad,rmsemul,rmsemuladd,rmsemulquad])}
table_rmse= pd.DataFrame(data)
table_rmse
#Out[301]: 
#         MODEL  ERROR_VALUES
#0     rmse_add    235.602674
#1     rmse_exp    268.693839
#2     rmse_lin    260.937814
#3    rmse_quad    297.406710
#4   rmseaddlin    135.553596
#5  rmseaddquad    218.193876
#6      rmsemul    239.654321
#7   rmsemuladd    160.683329
#8  rmsemulquad    160.683329


# FINAL model is 

finalmodel =smf.ols('Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Plastic_Sales).fit()
predictfinal=pd.Series(finalmodel.predict(Plastic_Sales))
predictfinal




# Therefore the ADDITIVE SEASONALITY WITH LINEAR TREND have the least mean squared error
#final model with least rmse value

predictfinal1=pd.Series(add_sealin.predict(Plastic_Sales))
predictfinal1
Plastic_Sales["Forecasted_Sales"]=pd.Series(predictfinal1)


##################################################################################
####################### D A T A   D R I V E N    M O D E L #######################
##################################################################################

# Converting the normal index of Plastic_Sales to time stamp 
plastic.index = pd.to_datetime(plastic.Month,format="%b-%y")

colnames = plastic.columns
colnames #Index(['Month', 'Sales', 'month'], dtype='object')

plastic["Sales"].plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction
#Amtrak["Day"] = Amtrak.Date.dt.strftime("%d") # Day extraction
#Amtrak["wkday"] = Amtrak.Date.dt.strftime("%A") # weekday extraction
plastic["year"] =plastic.Date.dt.strftime("%Y") # year extraction




# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Sales",data=plastic)
sns.boxplot(x="year",y="Sales",data=plastic)


# Line plot for Sales based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="month",data=plastic)


# moving average for the time series to understand better about the trend character in plastic
plastic.Sales.plot(label="org")
for i in range(2,24,6):
    plastic["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(plastic.Sales,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(plastic.Sales,lags=10)
tsa_plots.plot_pacf(plastic.Sales)


# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = plastic.head(48)
Test = plastic.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) # 17.041518935231107

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) #  101.99506639449883



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)#  21.051548547150357



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales) # 77.19731673846782



# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Sales"], label='Train',color="black")
plt.plot(Test.index, Test["Sales"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.legend(loc='best')

