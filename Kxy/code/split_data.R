####把数据集按收费口和出入方向进行区分
train = read.csv('E:\\KDD_CUP_2017\\dataSets\\dataSets\\training\\big_trainset_task2_version2.csv')
test = read.csv('E:\\KDD_CUP_2017\\dataSets\\dataSets\\testing_phase1\\big_testset_task2_version2.csv')
fix(test)
fix(train)
#names(train)
tollgate <- unique(train$tollgate_id)
direction <- unique(train$direction)
index_m <- merge(tollgate,direction)[-5,]
name_mat <- c("new_tollgate1_0.csv","new_tollgate2_0.csv","new_tollgate3_0.csv","new_tollgate1_1.csv","new_tollgate3_1.csv")
train <- subset(train,as.Date(substr(train$time_window,2,11)) < "2016-10-01" | "2016-10-07" < as.Date(substr(train$time_window,2,11)))  #去掉国庆节的数据

#################
##将前两个小时的车流量加到每个时间窗口后面
sr <- function(tollgate){
  self_regreession <- matrix(0,nrow = nrow(tollgate),ncol = 6)
  for (i in seq(7,nrow(tollgate),by = 6)){
    if (i>(nrow(tollgate)-5)){
      i <- nrow(tollgate)-5
    }
    for(j in 1:6){
      self_regreession[i:(i+5),j] <- tollgate$volume[i-7+j]
    }
  }
  return(cbind(tollgate,self_regreession))
}

for(i in 1:5){
  tollgate_split <- subset(train,train$tollgate_id == index_m[i,1] & train$direction == index_m[i,2]) 
  new_tollgate_split <- subset(tollgate_split,tollgate_split$hour__8 == 1|tollgate_split$hour__9 == 1|tollgate_split$hour__18 == 1|tollgate_split$hour__19 == 1)
  new_tollgate_split <- sr(new_tollgate_split)[-c(12:19,22:29,32:35,39)]#添加前两个小时的车流量同时把代表其他小时的列去掉
  path_t <- "E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\"
  path_n <- paste(path_t,name_mat[i],sep = '')
  write.csv(new_tollgate_split,file = path_n,row.names = F)
}
fix(new_tollgate_split)

# names(new_tollgate_split)
############将所有数据整合到一起
new_tollgate_split <- subset(train,train$hour__8 == 1|train$hour__9 == 1|train$hour__18 == 1|train$hour__19 == 1)
new_tollgate <- sr(new_tollgate_split)[-c(12:19,22:29,32:35,39)]#添加前两个小时的车流量同时把代表其他小时的列去掉
write.csv(new_tollgate,file = "E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\new_train_all.csv",row.names = F)
#对1_0进行微调，删去一些峰值，这些峰值对模型来说就是异常值，致使预测效果较差，删除后精度明显提高
train1_0 = read.csv('E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\new_tollgate1_0.csv')
train1_0 = subset(train1_0,train1_0$volume < 80)
write.csv(train1_0,file = "E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\new1_tollgate1_0.csv",row.names = F)
#对2_0进行微调，删去9月28号数据，该日数据不存在周期性，与其他数据差异较大
train2_0 = read.csv('E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\new_tollgate2_0.csv')
train2_0 = subset(train2_0,as.Date(substr(train2_0$time_window,2,11)) < "2016-09-28" | "2016-09-28" < as.Date(substr(train2_0$time_window,2,11)) )
write.csv(train2_0,file = 'E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\new1_tollgate2_0.csv',row.names = F)
#############################################################################################
#对test进行处理

for(i in 1:5){
  tollgate_split <- subset(test,test$tollgate_id == index_m[i,1] & test$direction == index_m[i,2])[,-c(11:18,21:28,31:34,38)]
  path_t <- "E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\"
  path_n <- paste(path_t,name_mat[i],sep = '')
  train <- read.csv(file=path_n)
  dim(tollgate_split)
  test_all <- cbind(tollgate_split,train[(nrow(train)-nrow(tollgate_split)+1):nrow(train),c('X1','X2','X3','X4','X5','X6')])
  path_ts <- "E:\\KDD_CUP_2017\\dataSets\\dataSets\\test_kxy\\"
  path_ns <- paste(path_ts,name_mat[i],sep = '')
  write.csv(test_all,file = path_ns,row.names = F)
}
  

# a <- train[(nrow(train)-nrow(tollgate_split)+1):nrow(train),c('X1','X2','X3','X4','X5','X6')]
# dim(a)



