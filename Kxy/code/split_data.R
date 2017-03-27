####把数据集按收费口和出入方向进行区分
train = read.csv('E:\\KDD_CUP_2017\\dataSets\\dataSets\\training\\big_trainset_task2_version2.csv')
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
      self_regreession[i,j] <- tollgate$volume[i-7+j]
      self_regreession[(i+1),j] <- tollgate$volume[i-7+j]
      self_regreession[(i+2),j] <- tollgate$volume[i-7+j]
      self_regreession[(i+3),j] <- tollgate$volume[i-7+j]
      self_regreession[(i+4),j] <- tollgate$volume[i-7+j]
      self_regreession[(i+5),j] <- tollgate$volume[i-7+j]
    }
  }
  return(cbind(tollgate,self_regreession))
}

for(i in 1:5){
  tollgate_split <- subset(train,train$tollgate_id == index_m[i,1] & train$direction == index_m[i,2]) 
  new_tollgate_split <- subset(tollgate_split,tollgate_split$hour__8 == 1|tollgate_split$hour__9 == 1|tollgate_split$hour__18 == 1|tollgate_split$hour__19 == 1)
  new_tollgate_split <- sr(new_tollgate_split)[-c(12:19,22:29,32:35)]#添加前两个小时的车流量同时把代表其他小时的列去掉
  path_t <- "E:\\KDD_CUP_2017\\dataSets\\dataSets\\train_kxy\\"
  path_n <- paste(path_t,name_mat[i],sep = '')
  write.csv(new_tollgate_split,file = path_n)
}
# fix(new_tollgate_split)
# names(new_tollgate_split)

