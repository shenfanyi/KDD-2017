# Task2: predicitons for traffic volume


setwd('.../Documents/KDD CUP 2017/team_work/3月24日数据_huakai/task2/original data')

data0 <- read.csv('big_trainset_task2_version2.csv')
test <- read.csv('big_testset_task2_version2.csv')

# training dataset: weekday & hour & X_span(minute):
data0$weekday[data0$Monday ==1] <- 1
data0$weekday[data0$Tuesday ==1] <- 2
data0$weekday[data0$Wednesday ==1] <- 3
data0$weekday[data0$Thursday ==1] <- 4
data0$weekday[data0$Friday ==1] <- 5
data0$weekday[data0$Saturday ==1] <- 6
data0$weekday[data0$Sunday ==1] <- 7

data0$hour[data0$hour__0 ==1] <- 0
data0$hour[data0$hour__1 ==1] <- 1
data0$hour[data0$hour__2 ==1] <- 2
data0$hour[data0$hour__3 ==1] <- 3
data0$hour[data0$hour__4 ==1] <- 4
data0$hour[data0$hour__5 ==1] <- 5
data0$hour[data0$hour__6 ==1] <- 6
data0$hour[data0$hour__7 ==1] <- 7
data0$hour[data0$hour__8 ==1] <- 8
data0$hour[data0$hour__9 ==1] <- 9
data0$hour[data0$hour__10 ==1] <- 10
data0$hour[data0$hour__11 ==1] <- 11
data0$hour[data0$hour__12 ==1] <- 12
data0$hour[data0$hour__13 ==1] <- 13
data0$hour[data0$hour__14 ==1] <- 14
data0$hour[data0$hour__15 ==1] <- 15
data0$hour[data0$hour__16 ==1] <- 16
data0$hour[data0$hour__17 ==1] <- 17
data0$hour[data0$hour__18 ==1] <- 18
data0$hour[data0$hour__19 ==1] <- 19
data0$hour[data0$hour__20 ==1] <- 20
data0$hour[data0$hour__21 ==1] <- 21
data0$hour[data0$hour__22 ==1] <- 22
data0$hour[data0$hour__23 ==1] <- 23

data0$X_span[data0$X0 == 1]  <- 'X0'
data0$X_span[data0$X20 == 1] <- 'X20'
data0$X_span[data0$X40 == 1] <- 'X40'

# test dataset:weekday & hour & X_span(minute):
test$weekday[test$Monday ==1] <- 1
test$weekday[test$Tuesday ==1] <- 2
test$weekday[test$Wednesday ==1] <- 3
test$weekday[test$Thursday ==1] <- 4
test$weekday[test$Friday ==1] <- 5
test$weekday[test$Saturday ==1] <- 6
test$weekday[test$Sunday ==1] <- 7

test$hour[test$hour__0 ==1] <- 0
test$hour[test$hour__1 ==1] <- 1
test$hour[test$hour__2 ==1] <- 2
test$hour[test$hour__3 ==1] <- 3
test$hour[test$hour__4 ==1] <- 4
test$hour[test$hour__5 ==1] <- 5
test$hour[test$hour__6 ==1] <- 6
test$hour[test$hour__7 ==1] <- 7
test$hour[test$hour__8 ==1] <- 8
test$hour[test$hour__9 ==1] <- 9
test$hour[test$hour__10 ==1] <- 10
test$hour[test$hour__11 ==1] <- 11
test$hour[test$hour__12 ==1] <- 12
test$hour[test$hour__13 ==1] <- 13
test$hour[test$hour__14 ==1] <- 14
test$hour[test$hour__15 ==1] <- 15
test$hour[test$hour__16 ==1] <- 16
test$hour[test$hour__17 ==1] <- 17
test$hour[test$hour__18 ==1] <- 18
test$hour[test$hour__19 ==1] <- 19
test$hour[test$hour__20 ==1] <- 20
test$hour[test$hour__21 ==1] <- 21
test$hour[test$hour__22 ==1] <- 22
test$hour[test$hour__23 ==1] <- 23

test$X_span[test$X0 == 1]  <- 'X0'
test$X_span[test$X20 == 1] <- 'X20'
test$X_span[test$X40 == 1] <- 'X40'

# Remove holiday data of 9.30~10.7: row3834 ~ row6562
remove_holidays <- c(3834:6562)
data1 <- data0[-remove_holidays,]

# prepare 5 training datasets:
tollgate1_entry <- subset(data1, tollgate_id == 1 & direction ==0)
tollgate1_exit <-  subset(data1, tollgate_id == 1 & direction ==1)
tollgate2_entry <- subset(data1, tollgate_id == 2 & direction ==0)
tollgate3_entry <- subset(data1, tollgate_id == 3 & direction ==0)
tollgate3_exit <-  subset(data1, tollgate_id == 3 & direction ==1)

# prepare the test submission dataset:
features <- c('tollgate_id','direction','time_window','weekday','hour','X_span')
test_submit <- test[features]
test_submit$volume <- 0

# predictions for 5 training datasets:
# 1）tollgate1_entry：
# subsect：volume, weekday, hour,X_span
features <- c('weekday','hour','X_span','volume')
tollgate1_entry <- subset(tollgate1_entry[features], hour == 8 |hour == 9 |hour ==17 |hour == 18)

# tapply: caculate Mon ~ Sun, 8~10am & 7~9pm, X0,X20,X40 median：
volume <-tapply(tollgate1_entry$volume, list(tollgate1_entry$weekday, tollgate1_entry$hour, tollgate1_entry$X_span), median)
hour_arr<-c(8,9,17,18)
x_span_arr<-c('X0','X20','X40')
for(i in seq(from=1,to=7,by=1)) 
  for(j in seq(from=1,to=4,by=1)) 
    for(k in seq(from=1,to=3,by=1)) {
      test_submit$volume[test_submit$tollgate_id == 1 & test_submit$direction ==0 & test_submit$weekday == i & test_submit$hour==hour_arr[j] & test_submit$X_span ==x_span_arr[k] ]<-volume[i,j,k]
    }

# 2）tollgate1_exit：
tollgate1_exit <- subset(tollgate1_exit[features], hour == 8 |hour == 9 |hour ==17 |hour == 18)
volume <-tapply(tollgate1_exit$volume, list(tollgate1_exit$weekday, tollgate1_exit$hour, tollgate1_exit$X_span), median)
for(i in seq(from=1,to=7,by=1)) 
  for(j in seq(from=1,to=4,by=1)) 
    for(k in seq(from=1,to=3,by=1)) {
      test_submit$volume[test_submit$tollgate_id == 1 & test_submit$direction ==1 & test_submit$weekday == i & test_submit$hour==hour_arr[j] & test_submit$X_span ==x_span_arr[k] ]<-volume[i,j,k]
    }

# 3）tollgate2_entry：
tollgate2_entry <- subset(tollgate2_entry[features], hour == 8 |hour == 9 |hour ==17 |hour == 18)
volume <-tapply(tollgate2_entry$volume, list(tollgate2_entry$weekday, tollgate2_entry$hour, tollgate2_entry$X_span), median)
for(i in seq(from=1,to=7,by=1)) 
  for(j in seq(from=1,to=4,by=1)) 
    for(k in seq(from=1,to=3,by=1)) {
      test_submit$volume[test_submit$tollgate_id == 2 & test_submit$direction ==0 & test_submit$weekday == i & test_submit$hour==hour_arr[j] & test_submit$X_span ==x_span_arr[k] ]<-volume[i,j,k]
    }

# 4）tollgate3_entry:
tollgate3_entry <- subset(tollgate3_entry[features], hour == 8 |hour == 9 |hour ==17 |hour == 18)
volume <-tapply(tollgate3_entry$volume, list(tollgate3_entry$weekday, tollgate3_entry$hour, tollgate3_entry$X_span), median)
for(i in seq(from=1,to=7,by=1)) 
  for(j in seq(from=1,to=4,by=1)) 
    for(k in seq(from=1,to=3,by=1)) {
      test_submit$volume[test_submit$tollgate_id == 3 & test_submit$direction ==0 & test_submit$weekday == i & test_submit$hour==hour_arr[j] & test_submit$X_span ==x_span_arr[k] ]<-volume[i,j,k]
    }

# 5）tollgate3_exit:
tollgate3_exit <- subset(tollgate3_exit[features], hour == 8 |hour == 9 |hour ==17 |hour == 18)
volume <-tapply(tollgate3_exit$volume, list(tollgate3_exit$weekday, tollgate3_exit$hour, tollgate3_exit$X_span), median)
for(i in seq(from=1,to=7,by=1)) 
  for(j in seq(from=1,to=4,by=1)) 
    for(k in seq(from=1,to=3,by=1)) {
      test_submit$volume[test_submit$tollgate_id == 3 & test_submit$direction ==1 & test_submit$weekday == i & test_submit$hour==hour_arr[j] & test_submit$X_span ==x_span_arr[k] ]<-volume[i,j,k]
    }

# mape result is 0.186
# *alternative approach: using 'mean' instead of 'median':mape result is 0.185

