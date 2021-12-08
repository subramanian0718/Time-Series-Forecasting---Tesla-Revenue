library(forecast)
library(zoo)
#setwd("C:/Users/STSC/Documents/Time Series/Project")
setwd("C:/Users/karth/Documents/673_Time_series/TESLA")

tesla.data <- read.csv("Tesla Historical Revenue (Quarterly) Data.csv")
length(tesla.data$Revenue)

head(tesla.data)
## Convert data to ts
tesla.ts <- ts(tesla.data$Revenue, 
                   start = c(2008, 4), end = c(2021, 1), freq = 4)
head(tesla.ts)
length(tesla.ts)

#########################################################################

plot(tesla.ts, 
     xlab = "Time", ylab = "Revenue (in millions)", 
      main = "Quarterly Revenue", col = "blue",yaxt="none",xaxt="none")

axis(1,seq(2008,2021,2))
axis(2,seq(50,10000,100))
########################################################################
tesla.stl <- stl(tesla.ts, s.window = "periodic")
autoplot(tesla.stl, main = "Time Series Components Of Quarterly Revenue Data")

##############################################################################
autocor <- Acf(tesla.ts, lag.max = 8, main = "Autocorrelation for Quarterly Revenue")
##############################################################
# Data partition
nValid <- 14
nTrain <- length(tesla.ts) - nValid + 3
nTrain
train.ts <- window(tesla.ts, start = c(2008, 4), end = c(2008, nTrain))
length(train.ts)
train.ts
valid.ts <- window(tesla.ts, start = c(2008, nTrain + 1), 
                   end = c(2008, nTrain + nValid))

head(train.ts)
length(train.ts)

head(valid.ts)
length(valid.ts)
valid.ts
##############################################################
#a trailing moving average 
ma.trailing_2 <- rollmean(train.ts, k = 2, align = "right")
ma.trailing_3 <- rollmean(train.ts, k = 3, align = "right")
ma.trailing_4 <- rollmean(train.ts, k = 4, align = "right")

ma.trailing_2.pred <- forecast(ma.trailing_2, h = nValid, level = 0)
ma.trailing_3.pred <- forecast(ma.trailing_3, h = nValid, level = 0)
ma.trailing_4.pred <- forecast(ma.trailing_4, h = nValid, level = 0)

round(accuracy(ma.trailing_2.pred$mean, valid.ts), 3)
round(accuracy(ma.trailing_3.pred$mean, valid.ts), 3)
round(accuracy(ma.trailing_4.pred$mean, valid.ts), 3)
################################################################
#b.	Simple and Advanced Exponential Smoothing
## SIMPLE EXPONENTIAL SMOOTHING WITH ORIGINAL DATA AND OPTIMAL ALPHA.
ses.tesla <- ets(train.ts, model = "ANN")
ses.tesla

ses.tesla.pred <- forecast(ses.tesla, h = nValid, level = 0)
ses.tesla.pred

round(accuracy(ses.tesla.pred$mean, valid.ts), 3)
round(accuracy(ses.tesla$fitted, train.ts), 3)

###################################################
# Holt's model with optimal smoothing parameters.
tesla.holts.opt <- ets(train.ts, model = "ZZN")
tesla.holts.opt
tesla.holts.opt.pred <- forecast(tesla.holts.opt, h = nValid, level = 0)
tesla.holts.opt.pred
round(accuracy(tesla.holts.opt.pred$mean, valid.ts), 3)
#####################################################
# Holt-Winter's (HW) exponential smoothing 
tesla.holts.w.opt <- ets(train.ts, model = "ZZZ")
tesla.holts.w.opt
tesla.holts.w.opt.pred <- forecast(tesla.holts.w.opt, h = nValid, level = 0)
tesla.holts.w.opt.pred
round(accuracy(tesla.holts.w.opt.pred$mean, valid.ts), 3)
############################################################
#c.Multiple Regression Models with Trend and Seasonality
# Linear trend model 
tesla.trend <- tslm(train.ts ~ trend)
summary(tesla.trend)
tesla.trend.pred <- forecast(tesla.trend, h = nValid, level = 0)
round(accuracy(tesla.trend.pred$mean, valid.ts), 3)
###################################################
#Linear trend with seasonality 
tesla.trend.seas <-tslm(train.ts ~ trend + season)
summary(tesla.trend.seas)
tesla.trend.seas.pred <- forecast(tesla.trend.seas, h = nValid, level = 0)
round(accuracy(tesla.trend.seas.pred$mean, valid.ts), 3)
###########################################################
#Quadratic trend 
tesla.quad.trend <- tslm(train.ts ~ trend + I(trend^2) )
summary(tesla.quad.trend)
tesla.quad.trend.pred <- forecast(tesla.quad.trend , h = nValid, level = 0)
round(accuracy(tesla.quad.trend.pred$mean, valid.ts), 3)
########################################################
#Quadratic trend with seasonality 
tesla.quad.seas.trend <- tslm(train.ts ~ trend + I(trend^2) + season)
summary(tesla.quad.seas.trend)
tesla.quad.seas.trend.pred <- forecast(tesla.quad.seas.trend , h = nValid, level = 0)
round(accuracy(tesla.quad.seas.trend.pred$mean, valid.ts), 3)
#############################################################
trainq.trend.season <- tslm(train.ts ~ trend + I(trend^2) + season)
trainq.trend.season.pred <- forecast(trainq.trend.season, h = nValid, level = 0)
#####################################################
## Two level quadractic trend and seasonality +AR(1)
res.ar1 <- Arima(trainq.trend.season$residuals, order = c(1,0,0))
summary(res.ar1)
res.ar1.pred <- forecast(res.ar1, h = nValid, level = 0)
res.ar1.pred

Acf(res.ar1$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

valid.two.level <- trainq.trend.season.pred$mean + res.ar1.pred$mean
summary(valid.two.level)

valid.two.level.pred <- forecast(valid.two.level, h = nValid, level = 0)

round(accuracy(trainq.trend.season.pred$mean + res.ar1.pred$mean, valid.ts), 3)
####################################################################################
## Two level quadractic trend  +AR(1)
res1.ar1 <- Arima(tesla.quad.trend$residuals, order = c(1,0,0))
summary(res1.ar1)
res1.ar1.pred <- forecast(res1.ar1, h = nValid, level = 0)
res1.ar1.pred

Acf(res1.ar1$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

valid.two.level <- tesla.quad.trend.pred$mean + res1.ar1.pred$mean
summary(valid.two.level)

valid.two.level.pred <- forecast(valid.two.level, h = nValid, level = 0)

round(accuracy(tesla.quad.trend.pred$mean + res1.ar1.pred$mean, valid.ts), 3)


###########################################################################
# Fit a regression model with quadratic trend and seasonality + Trailing MA (2 LEVEL)
# training partition. 
trend.seas <- tslm(train.ts ~ trend + I(trend^2) + season)
summary(trend.seas)

# Identify and display residuals for time series based on the regression
# (differences between actual and regression values in the same periods).
trend.seas.res <- trend.seas$residuals
trend.seas.res

# Apply trailing MA for residuals with window width k = 4. 
ma.trail.res <- rollmean(trend.seas.res, k = 4, align = "right")
ma.trail.res

# Create regression forecast with trend and seasonality for 
# validation period.
trend.seas.pred <- forecast(trend.seas, h = nValid, level = 0)
trend.seas.pred

# Regression residuals in validation period.
trend.seas.res.valid <- valid.ts - trend.seas.pred$mean
trend.seas.res.valid

# Create residuals forecast for validation period.
ma.trail.res.pred <- forecast(ma.trail.res, h = nValid, level = 0)
ma.trail.res.pred$mean


# Develop two-level forecast for validation period by combining  
# regression forecast and trailing MA forecast for residuals.
fst.2level <- trend.seas.pred$mean + ma.trail.res.pred$mean
fst.2level

# Create a table for validation period: validation data, regression 
# forecast, trailing MA for residuals and total forecast.
valid.df <- data.frame(valid.ts, trend.seas.pred$mean, 
                       ma.trail.res.pred$mean, 
                       fst.2level)
names(valid.df) <- c("Tesla Revenue", "Regression.Fst", 
                     "MA.Residuals.Fst", "Combined.Fst")
valid.df

# Use accuracy() function to identify common accuracy measures.
# Use round() function to round accuracy measures to three decimal digits.
round(accuracy(fst.2level, valid.ts), 3)

########################################################


############################################################
## FIT AUTO ARIMA MODEL.

# Use auto.arima() function to fit ARIMA model.
# Use summary() to show auto ARIMA model and its parameters.
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model in validation set.  
train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Plot ts data, trend and seasonality data, and predictions for validation period.
plot(train.auto.arima.pred, 
     xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(0, 12000), bty = "l",
     xaxt = "n", xlim = c(2008, 2024), 
     main = "Auto ARIMA Model", lwd = 2, flty = 5) 
axis(1, at = seq(2008, 2024, 1), labels = format(seq(2008, 2024, 1)))
lines(train.auto.arima.pred$fitted, col = "blue", lwd = 2)
lines(valid.ts, col = "black", lwd = 2, lty = 1)
legend(2008,4000, legend = c("Tesla Revenue Time Series", 
                             "Auto ARIMA Forecast for Training Period",
                             "Auto ARIMA Forecast for Validation Period"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# Plot on chart vertical lines and horizontal arrows describing
# training, validation, and future prediction intervals.
lines(c(2017.5,2017.50), c(0, 12000))
lines(c(2021, 2021), c(0, 12000))
text(2012.25, 11700, "Training")
text(2018.75, 11700, "Validation")
text(2021.75, 11700, "Future")
arrows(2017.25, 11000, 2008, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2017.75, 11000, 2021, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 11000, 2023.75, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Using Acf() function, create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(train.auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

#Accuracy function in training data set
round(accuracy(train.auto.arima$fitted, train.ts), 3)
#Accuracy function in validation data set
round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)

##############################################################
## FIT ARIMA(2,1,2)(1,1,0) MODEL.

# Use Arima() function to fit ARIMA(2,1,2)(1,1,2) model for 
# trend and seasonality.
# Use summary() to show ARIMA model and its parameters.
train.arima.seas <- Arima(train.ts, order = c(2,1,2), 
                          seasonal = c(1,1,0)) 
summary(train.arima.seas)

# Apply forecast() function to make predictions for ts with 
# ARIMA model in validation set.    
train.arima.seas.pred <- forecast(train.arima.seas, h = nValid, level = 0)
train.arima.seas.pred


# Plot ts data, ARIMA model, and predictions for validation period.
plot(train.arima.seas.pred, 
     xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(0, 12000), bty = "l",
     xaxt = "n", xlim = c(2008, 2024), 
     main = "ARIMA(2,1,2)(1,1,0)[4] Model", lwd = 2, flty = 5) 
axis(1, at = seq(2008, 2024, 1), labels = format(seq(2008, 2024, 1)))
lines(train.arima.seas.pred$fitted, col = "blue", lwd = 2)
lines(valid.ts, col = "black", lwd = 2, lty = 1)
legend(2008,8000, legend = c("Tesla Revenue Time Series", 
                             "Seasonal ARIMA Forecast for Training Period",
                             "Seasonal ARIMA Forecast for Validation Period"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")


# Plot on chart vertical lines and horizontal arrows describing
# training, validation, and future prediction intervals.
lines(c(2017.5,2017.50), c(0, 12000))
lines(c(2021, 2021), c(0, 12000))
text(2012.25, 11700, "Training")
text(2018.75, 11700, "Validation")
text(2021.75, 11700, "Future")
arrows(2017.25, 11000, 2008, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2017.75, 11000, 2021, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 11000, 2023.75, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Use Acf() function to create autocorrelation chart of ARIMA(2,1,2)(1,1,2) 
# model residuals.
Acf(train.arima.seas$residuals, lag.max = 12, 
    main = "Autocorrelations of ARIMA(2,1,2)(1,1,2) Model Residuals")

#Accuracy function in training data set
round(accuracy(train.arima.seas$fitted, train.ts), 3)

#Accuracy function in validation data set
round(accuracy(train.arima.seas.pred$mean, valid.ts), 3)



#########################################################
#All training & validation data sets accuracy for other Models

#Trailing MA WIDTH 2
round(accuracy(ma.trailing_2.pred$mean, valid.ts), 3)
#Trailing MA WIDTH 3
round(accuracy(ma.trailing_3.pred$mean, valid.ts), 3)
#Trailing MA WIDTH 4
round(accuracy(ma.trailing_4.pred$mean, valid.ts), 3)

# SIMPLE EXPONENTIAL SMOOTHING 
round(accuracy(ses.tesla.pred$mean, valid.ts), 3)

#HOLTS MODEL
round(accuracy(tesla.holts.opt.pred$mean, valid.ts), 3)

#HOLTS WINTER MODEL
round(accuracy(tesla.holts.w.opt.pred$mean, valid.ts), 3)

# Linear trend model
round(accuracy(tesla.trend.pred$mean, valid.ts), 3)

#Linear trend with seasonality 
round(accuracy(tesla.trend.seas.pred$mean, valid.ts), 3)

#Quadratic trend          #
round(accuracy(tesla.quad.trend.pred$mean, valid.ts), 3)

#Quadratic trend with seasonality 
round(accuracy(tesla.quad.seas.trend.pred$mean, valid.ts), 3)

## Two level Model with Regression Quadractic trend and seasonality +AR(1) model
round(accuracy(trainq.trend.season.pred$mean + res.ar1.pred$mean, valid.ts), 3)	

## Two level Model with Regression Quadractic  trend  +AR(1) model
round(accuracy(tesla.quad.trend.pred$mean + res1.ar1.pred$mean, valid.ts), 3)

#Two level Model with Regression Quadractic  trend and seasonality +Trailing MA(1)
round(accuracy(fst.2level, valid.ts), 3)

#Auto ARIMA model.
round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)

#ARIMA(2,1,2)(1,1,2) model;
round(accuracy(train.arima.seas.pred$mean, valid.ts), 3)


#############################################################
##########################################################

#Entire data set analysis

###########################################################
#############################################################
##Quadratic trend + Seasonality with AR(1) FOR ENTIRE DATA SET

tesla.total.quad.seas.trend <- tslm(tesla.ts ~ trend + I(trend^2) + season)
summary(tesla.total.quad.seas.trend)
tesla.total.quad.seas.trend.pred <- forecast(tesla.total.quad.seas.trend , h = 4,level=0)
round(accuracy(tesla.total.quad.seas.trend.pred$fitted, valid.ts), 3)
 


Acf(tesla.total.quad.seas.trend$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

res.total.ar1 <- Arima(tesla.total.quad.seas.trend$residuals, order = c(1,0,0))
summary(res.total.ar1)
res.total.ar1.pred <- forecast(res.total.ar1, h = 4, level = 0)
res.total.ar1.pred

Acf(res.total.ar1$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

valid.total.two.level <- tesla.total.quad.seas.trend.pred$mean + res.total.ar1.pred$mean
summary(valid.total.two.level)

#valid.two.total.level.pred <- forecast(valid.total.two.level, h = 4, level = 0)

#valid.two.total.level.pred

round(accuracy(tesla.total.quad.seas.trend.pred$fitted + res.total.ar1.pred$fitted, tesla.ts), 3)

####################################################################################
#SES for entire data set
## SIMPLE EXPONENTIAL SMOOTHING (SES) WITH ORIGINAL DATA, ALPHA = 0.2.


## SIMPLE EXPONENTIAL SMOOTHING WITH ORIGINAL DATA AND OPTIMAL ALPHA.

# Create simple exponential smoothing (SES) for Amtrak data with optimal alpha.
# Use ets() function with model = "ANN", i.e., additive error(A), no trend (N),
# & no seasonality (N). Use optimal alpha to fit SES over the original data.
ses.opt <- ets(tesla.ts, model = "ANN")
ses.opt

# Use forecast() function to make predictions using this SES model with optimal alpha
# and 12 periods into the future.
# Show predictions in tabular format
ses.opt.pred <- forecast(ses.opt, h = 4, level = 0)
ses.opt.pred


# Plot ses predictions for original data and optimal alpha.
plot(ses.opt.pred, 
     xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(0, 12000), bty = "l",
     xaxt = "n", xlim = c(2008, 2024), lwd = 2,
     main = "Original Data and SES Optimal Forecast, Alpha = 0.9999", 
     flty = 5) 
axis(1, at = seq(2008, 2024, 1), labels = format(seq(2008, 2024, 1)))
lines(ses.opt.pred$fitted, col = "blue", lwd = 2)

# Plot on chart vertical lines and horizontal arrows describing
# training, validation, and future prediction intervals.

lines(c(2021, 2021), c(0, 12000))
text(2012.25, 11700, "Training")
text(2022.5, 11700, "Future")
arrows(2021, 11000, 2008, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.5, 11000, 2023.75, 11000, code = 3, length = 0.1,
       lwd = 1, angle = 30)



##  ACCURACY OF THE TWO SES WITH  OPTIMAL ALPHA. 
round(accuracy(ses.opt.pred$fitted, tesla.ts),3)


########################################################################################
#HW for entire data set
## FORECAST WITH HOLT-WINTER'S MODEL USING ENTIRE DATA SET INTO
## THE FUTURE FOR 12 PERIODS.

# Create Holt-Winter's (HW) exponential smoothing for full Amtrak data set. 
# Use ets() function with model = "ZZZ", to identify the best HW option
# and optimal alpha, beta, & gamma to fit HW for the entire data period.
HW.ZZZ <- ets(tesla.ts, model = "ZZZ")
HW.ZZZ # Model appears to be (A, N, A), with alpha = 0.5558 and gamma = 0.0003.

# Use forecast() function to make predictions using this HW model for
# 12 month into the future.
HW.ZZZ.pred <- forecast(HW.ZZZ, h = 4 , level = 95)
HW.ZZZ.pred


# Identify performance measures for HW forecast.
round(accuracy(HW.ZZZ.pred$fitted, tesla.ts), 3)

###############################################################3
#Quadratic trend + SEASONALITY for entire data set

## FIT REGRESSION MODEL WITH QUADRATIC TREND AND SEASONALITY 
## FOR ENTIRE DATASET. FORECAST AND PLOT DATA, AND MEASURE ACCURACY.

# Use tslm() function to create quadratic trend and seasonality model.
trend.season <- tslm(tesla.ts ~ trend + I(trend^2) + season)

# See summary of quadratic trend and seasonality equation and associated parameters.
summary(trend.season)

# Apply forecast() function to make predictions for ts with 
# trend and seasonality data in 12 future periods.
trend.season.pred <- forecast(trend.season, h = 4, level = 0)


# Use accuracy() function to identify common accuracy measures
# for naive model, seasonal naive, and regression model with quadratic trend and seasonality.
round(accuracy(trend.season.pred$fitted, tesla.ts),3)

######################################################################
#Quadratic trend  for entire data set

## FIT REGRESSION MODEL WITH QUADRATIC TREND AND SEASONALITY 
## FOR ENTIRE DATASET. FORECAST AND PLOT DATA, AND MEASURE ACCURACY.

# Use tslm() function to create quadratic trend and seasonality model.
trend.quad <- tslm(tesla.ts ~ trend + I(trend^2) )

# See summary of quadratic trend and seasonality equation and associated parameters.
summary(trend.quad)

# Apply forecast() function to make predictions for ts with 
# trend and seasonality data in 12 future periods.
trend.quad.pred <- forecast(trend.quad, h = 12, level = 0)


# Use accuracy() function to identify common accuracy measures
# for naive model, seasonal naive, and regression model with quadratic trend and seasonality.
round(accuracy(trend.quad.pred$fitted, tesla.ts),3)

############################################################################

##Quadratic trend with AR(1) FOR ENTIRE DATA SET

tesla.total.quad.trend <- tslm(tesla.ts ~ trend + I(trend^2))
summary(tesla.total.quad.trend)
tesla.total.quad.trend.pred <- forecast(tesla.total.quad.trend , h = 4,level=0)
round(accuracy(tesla.total.quad.trend.pred$fitted, valid.ts), 3)



Acf(tesla.total.quad.trend$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

res.total.ar1 <- Arima(tesla.total.quad.trend$residuals, order = c(1,0,0))
summary(res.total.ar1)
res.total.ar1.pred <- forecast(res.total.ar1, h = 4, level = 0)
res.total.ar1.pred

Acf(res.total.ar1$residuals, lag.max = 8, 
    main = "Autocorrelation for Tesla Revenue Training Residuals of Residuals")

valid.total.two.level <- tesla.total.quad.trend.pred$mean + res.total.ar1.pred$mean
summary(valid.total.two.level)

#valid.two.total.level.pred <- forecast(valid.total.two.level, h = 4, level = 0)

#valid.two.total.level.pred

round(accuracy(tesla.total.quad.trend.pred$fitted + res.total.ar1.pred$fitted, tesla.ts), 3)
##################################################################################
# Fit a regression model with quadratic trend and seasonality +Trailing MA
# entire data set.
tot.trend.seas <- tslm(tesla.ts ~ trend + I(trend^2) + season)
summary(tot.trend.seas)

# Create regression forecast for future 12 periods.
tot.trend.seas.pred <- forecast(tot.trend.seas, h = 4, level = 0)
tot.trend.seas.pred

# Identify and display regression residuals for entire data set.
tot.trend.seas.res <- tot.trend.seas$residuals
tot.trend.seas.res

# Use trailing MA to forecast residuals for entire data set.
tot.ma.trail.res <- rollmean(tot.trend.seas.res, k = 4, align = "right")
tot.ma.trail.res

# Create forecast for trailing MA residuals for future 12 periods.
tot.ma.trail.res.pred <- forecast(tot.ma.trail.res, h = 4, level = 0)
tot.ma.trail.res.pred

# Develop 2-level forecast for future 12 periods by combining 
# regression forecast and trailing MA for residuals for future
# 12 periods.
tot.fst.2level <- tot.trend.seas.pred$mean + tot.ma.trail.res.pred$mean
tot.fst.2level

# Create a table with regression forecast, trailing MA for residuals,
# and total forecast for future 12 periods.
future12.df <- data.frame(tot.trend.seas.pred$mean, tot.ma.trail.res.pred$mean, 
                          tot.fst.2level)
names(future12.df) <- c("Regression.Fst", "MA.Residuals.Fst", "Combined.Fst")
future12.df

# Use accuracy() function to identify common accuracy measures.

round(accuracy(tot.trend.seas.pred$fitted+tot.ma.trail.res, tesla.ts), 3)


#############################################################################
## FIT SEASONAL ARIMA FOR ENTIRE DATA SET. 

# Use arima() function to fit seasonal ARIMA(2,1,2)(1,1,0) model 
# for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
arima.seas <- Arima(tesla.ts, order = c(2,1,2),seasonal = c(1,1,0)) 
summary(arima.seas)

# Apply forecast() function to make predictions for ts with 
# seasonal ARIMA model for the future 12 periods. 
arima.seas.pred <- forecast(arima.seas, h = 4, level = 0)
arima.seas.pred
arima.seas.pred$mean
arima.seas$fitted

# Plot historical data, predictions for historical data, and seasonal 
# ARIMA forecast for 12 future periods.
plot(tesla.ts, 
     xlab = "Time", ylab = "Tesla Revenue  (in Millions)", ylim = c(0, 13500), bty = "l",
     xaxt = "n", xlim = c(2008, 2024), lwd = 2,
     main = "ARIMA(2,1,2)(1,1,0)[4] Model for Entire Dataset") 
axis(1, at = seq(2008, 2024, 1), labels = format(seq(2008, 2024, 1)))
lines(arima.seas$fitted, col = "blue", lwd = 2)
lines(arima.seas.pred$mean, col = "blue", lty = 5, lwd = 2)
legend(2008,4000, legend = c("Tesla Revenue Series", 
                             "Seasonal ARIMA Forecast", 
                             "Seasonal ARIMA Forecast for 12 Future Periods"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# plot on the chart vertical lines and horizontal arrows
# describing training and future prediction intervals.
# lines(c(2004.25 - 3, 2004.25 - 3), c(0, 2600))
# Plot on chart vertical lines and horizontal arrows describing
# training, validation, and future prediction intervals.
lines(c(2021, 2021), c(0, 14000))
text(2012.25, 13700, "Training")
text(2022.25, 13700, "Future")
arrows(2008, 13300, 2021, 13300, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.25, 13300, 2023.75, 13300, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Use Acf() function to create autocorrelation chart of seasonal ARIMA 
# model residuals.
Acf(arima.seas$residuals, lag.max = 12, 
    main = "Autocorrelations of Seasonal ARIMA Model Residuals")

# (1) Seasonal ARIMA (2,1,2)(1,1,2) Model,
round(accuracy(arima.seas.pred$fitted, tesla.ts), 3)

#########################################################################
## FIT AUTO ARIMA MODELS FOR ENTIRE DATA SET. 

# Use auto.arima() function to fit ARIMA model for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
auto.arima <- auto.arima(tesla.ts)
summary(auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model for the future 4 periods. 
auto.arima.pred <- forecast(auto.arima, h = 4, level = 0)
auto.arima.pred


# Plot historical data, predictions for historical data, and Auto ARIMA 
# forecast for 12 future periods.
plot(tesla.ts, 
     xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(0, 13500), bty = "l",
     xaxt = "n", xlim = c(2008, 2024), lwd = 2,
     main = "Auto ARIMA Model for Entire Dataset") 
axis(1, at = seq(2008, 2024, 1), labels = format(seq(2008, 2024, 1)))
lines(auto.arima$fitted, col = "blue", lwd = 2)
lines(auto.arima.pred$mean, col = "blue", lty = 5, lwd = 2)
legend(2008,8000, legend = c("Tesla Revenue Series", 
                             "Auto ARIMA Forecast", 
                             "Auto ARIMA Forecast for 12 Future Periods"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# plot on the chart vertical lines and horizontal arrows
# describing training and future prediction intervals.
# lines(c(2004.25 - 3, 2004.25 - 3), c(0, 2600))
lines(c(2021, 2021), c(0, 14000))
text(2012.25, 13500, "Training")
text(2022.25, 13500, "Future")
arrows(2008, 13000, 2021, 13000, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2021.25, 13000, 2023.75, 13000, code = 3, length = 0.1,
       lwd = 1, angle = 30)


# Use Acf() function to create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

# MEASURE FORECAST ACCURACY FOR ENTIRE DATA SET.

# Use accuracy() function to identify common accuracy measures for:
# (1) Seasonal ARIMA (2,1,2)(1,1,2) Model,
# (1) Auto ARIMA Model,
round(accuracy(arima.seas.pred$fitted, tesla.ts), 3)
round(accuracy(auto.arima.pred$fitted, tesla.ts), 3)

#############################################################################
# Linear trend model 
tesla.lin.trend <- tslm(tesla.ts ~ trend)
summary(tesla.lin.trend)
tesla.lin.trend <- forecast(tesla.lin.trend, h = 4, level = 0)
round(accuracy(tesla.lin.trend$fitted, tesla.ts), 3)
###################################################
#Linear trend with seasonality 
tesla.lin.trend.seas <-tslm(tesla.ts ~ trend + season)
summary(tesla.lin.trend.seas)
tesla.trend.seas.pred <- forecast(tesla.lin.trend.seas, h = 4, level = 0)
round(accuracy(tesla.lin.trend.seas$fitted, tesla.ts), 3)


#########################################################################

#a
#trailing moving average 
ma.tesla.trailing_2 <- rollmean(tesla.ts, k = 2, align = "right")
ma.tesla.trailing_3 <- rollmean(tesla.ts, k = 3, align = "right")
ma.tesla.trailing_4 <- rollmean(tesla.ts, k = 4, align = "right")

ma.tesla.trailing_2.pred <- forecast(ma.tesla.trailing_2, h = 4, level = 0)
ma.tesla.trailing_3.pred <- forecast(ma.tesla.trailing_3, h = 4, level = 0)
ma.tesla.trailing_4.pred <- forecast(ma.tesla.trailing_4, h = 4, level = 0)

round(accuracy(ma.tesla.trailing_2.pred$fitted, tesla.ts), 3)
round(accuracy(ma.tesla.trailing_3.pred$fitted, tesla.ts), 3)
round(accuracy(ma.tesla.trailing_4.pred$fitted, tesla.ts), 3)

#####################################################################
####################################

#Entire data sets accuracy


## Performance measure for ACCURACY OF THE TWO SES WITH  OPTIMAL ALPHA. 
round(accuracy(ses.opt.pred$fitted, tesla.ts),3)

# Performance measures for HW forecast.
round(accuracy(HW.ZZZ.pred$fitted, tesla.ts), 3)

#Performance measures for Quadratic trend + SEASONALITY for entire data set
round(accuracy(trend.season.pred$fitted, tesla.ts),3)

#Performance measures for Quadratic trend  for entire data set
round(accuracy(trend.quad.pred$fitted, tesla.ts),3)

#Performance measures for Quadratic trend + Seasonality with AR(1) FOR ENTIRE DATA SET
round(accuracy(tesla.total.quad.seas.trend.pred$fitted + res.total.ar1.pred$fitted, tesla.ts), 3)

#Performance measures for Quadratic trend + with AR(1) FOR ENTIRE DATA SET
round(accuracy(tesla.total.quad.trend.pred$fitted + res.total.ar1.pred$fitted, tesla.ts), 3)

#Performance measures for Quadratic trend + TMA
round(accuracy(tot.trend.seas.pred$fitted+tot.ma.trail.res, tesla.ts), 3)

#Performance measures for Linear trend model 
round(accuracy(tesla.lin.trend$fitted, tesla.ts), 3)

#Performance measures for Linear trend with seasonality 
round(accuracy(tesla.lin.trend.seas$fitted, tesla.ts), 3)

#Performance measure for Trailing MA with width of 2
round(accuracy(ma.tesla.trailing_2.pred$fitted, tesla.ts), 3)

#Performance measure for Trailing MA with width of 3
round(accuracy(ma.tesla.trailing_3.pred$fitted, tesla.ts), 3)

#Performance measure for Trailing MA with width of 4
round(accuracy(ma.tesla.trailing_4.pred$fitted, tesla.ts), 3)

# Performance measure for Seasonal ARIMA (2,1,2)(1,1,2) Model,
round(accuracy(arima.seas.pred$fitted, tesla.ts), 3)

# Performance measure for Auto ARIMA Model,
round(accuracy(auto.arima.pred$fitted, tesla.ts), 3)

# Performance measure for Seasonal naive Model,
round(accuracy((snaive(tesla.ts))$fitted, tesla.ts), 3)

# Performance measure for NAive model
round(accuracy((naive(tesla.ts))$fitted, tesla.ts), 3)





















