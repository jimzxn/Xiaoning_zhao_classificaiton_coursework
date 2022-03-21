hotels <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv")
View(hotels)
set.seed(42)
table(hotels$hotel)
table(hotels$country)
table(hotels$meal)
table(hotels$market_segment)
table(hotels$reserved_room_type)
table(hotels$deposit_type)
table(hotels$reservation_status)
table(hotels$customer_type)
table(hotels$is_canceled)
table(hotels$reservation_status)
skimr::skim(hotels)
DataExplorer::plot_bar(hotels, ncol = 3)
DataExplorer::plot_histogram(hotels, ncol = 3)
DataExplorer::plot_boxplot(hotels, by = "is_canceled", ncol = 3)
install.packages("data.table")
install.packages("mlr3verse")
install.packages('cleandata')
library("data.table")
library("mlr3verse")
library('cleandata')
data<-data.frame(hotels)
data2<-as.data.frame(unclass(data),stringsAsFactors = TRUE)
data2[is.na(data2)] <- 0
data3<-sapply(data2,as.numeric)
data3<-data.frame(data3)
data4<-copy(data3)
data4$reservation_status<-NULL
data2$reservation_status_date<-as.numeric(data2$reservation_status_date)
#data$reservation_status_date<-as.POSIXct(data$reservation_status_date)
data3$is_canceled <- as.factor(data3$is_canceled)
data4<-copy(data3)
data4$reservation_status<-NULL

plot(data3$is_canceled, data3$reservation_status, pch = 19, col = "lightblue")
cor(as.numeric(data3$is_canceled), data3$reservation_status)

hotel_task <- TaskClassif$new(id = "hotels",backend = data2,target = "is_canceled",positive = '1')
hotel_task_numeric <- TaskClassif$new(id = "hotels",backend = data3,target = "is_canceled",positive = '1')
hotel_task_drop<-TaskClassif$new(id = "hotels_drop_one",backend = data4,target = "is_canceled",positive = '1')
cv5 <- rsmp("cv", folds = 5)
boot<-rsmp('bootstrap')
cv5D<-rsmp('cv',folds=5)
cv5$instantiate(hotel_task_numeric)
boot$instantiate(hotel_task_numeric)
cv5D$instantiate(hotel_task_drop)
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_tree <- lrn("classif.rpart", predict_type = "prob")
lrn_nb <- lrn("classif.naive_bayes", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrn_xgb <- lrn("classif.xgboost", predict_type = "prob")
lrn_knn <- lrn("classif.kknn", predict_type = "prob")
res <- benchmark(data.table(
  task       = list(hotel_task_numeric),
  learner    = list(lrn_baseline,
                    lrn_tree,
                    lrn_lr,
                    lrn_nb,
                    lrn_lda,
                    lrn_xgb,
                    lrn_knn
                    ),
  resampling = list(cv5)
), store_models = TRUE)

res2 <- benchmark(data.table(
  task       = list(hotel_task_numeric),
  learner    = list(lrn_baseline,
                    lrn_tree,
                    lrn_lr,
                    lrn_nb,
                    lrn_lda,
                    lrn_xgb,
                    lrn_knn
  ),
  resampling = list(boot)
), store_models = TRUE)
resD <- benchmark(data.table(
  task       = list(hotel_task_drop),
  learner    = list(lrn_baseline,
                    lrn_tree,
                    lrn_lr,
                    lrn_nb,
                    lrn_lda,
                    lrn_xgb,
                    lrn_knn
  ),
  resampling = list(cv5D)
), store_models = TRUE)
res$combine(res2)
res$combine(resD)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
prefo_plot<-resD$aggregate(list(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc"),
                               msr("classif.fpr"),
                               msr("classif.fnr")))
plot(factor(prefo_plot$learner_id),prefo_plot$classif.acc)
plot(factor(prefo_plot$learner_id),prefo_plot$classif.auc)

trees <- res$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

trees <- resD$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

cv3<-rsmp('cv',folds=3)
cv7<-rsmp('cv',folds=7)
cv3$instantiate(hotel_task_drop)
cv7$instantiate(hotel_task_drop)

reso <- benchmark(data.table(
  task       = list(hotel_task_drop),
  learner    = list(lrn_lda),
  resampling = list(cv3,cv5,cv7,boot)
), store_models = TRUE)

resolr <- benchmark(data.table(
  task       = list(hotel_task_drop),
  learner    = list(lrn_lr),
  resampling = list(cv3,cv5,cv7,boot)
), store_models = TRUE)

reso$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
resolr$aggregate(list(msr("classif.ce"),
                    msr("classif.acc"),
                    msr("classif.auc"),
                    msr("classif.fpr"),
                    msr("classif.fnr")))
reso$combine(resolr)
install.packages('mlr3tuning')
library('mlr3tuning')

at_lr<-AutoTuner$new(learner = lrn('classif.log_reg',epsilon=to_tune(1e-09, 1e-07, logscale = TRUE),predict_type = "prob"),resampling=rsmp('cv',folds=5)
                  ,measure = msr("classif.acc"),terminator = trm("evals", n_evals = 20),tuner = tnr("random_search"))
at_lr$train(hotel_task_drop)

# at_lda<-AutoTuner$new(learner = lrn('classif.lda',method=to_tune(c('moment', 'mle', 'mve', 't')),
#                                 predict.method=to_tune(c('plug-in', 'predictive', 'debiased')),predict_type = "prob"),resampling=rsmp('cv',folds=5)
#                   ,measure = msr("classif.acc"),terminator = trm("evals", n_evals = 3),tuner = tnr("random_search"))
# 
# at_lda$train(hotel_task_drop)



# library(keras)
# library(tensorflow)
# bound <- floor((nrow(data4)/4)*3)        
# data4 <- data4[sample(nrow(data4)), ]         
# train <- data4[1:bound, ]             
# test <- data4[(bound+1):nrow(data4), ]   
# 
# train_label<-drop(train$is_canceled)
# train_label<-data.frame(train_label)
# train$is_canceled<-NULL
# 
# test_label<-drop(test$is_canceled)
# test_label<-data.frame(test_label)
# test$is_canceled<-NULL
# 
# train<-as.matrix(train)
# train_label<-as.matrix(train_label)
# 
# test<-as.matrix(test)
# test_label<-as.matrix(test_label)
# 
# model <- keras_model_sequential()
# 
# model %>% 
#   layer_dense(units = 64, input_shape = 30) %>% 
#   layer_activation(activation = 'relu') %>%layer_dense(units = 32) %>% 
#   layer_activation(activation = 'relu') %>%layer_dense(units = 2) %>% 
#   layer_activation(activation = 'softmax')
# model %>% compile(loss = 'categorical_crossentropy',optimizer = optimizer_adam(learning_rate = 0.1),metrics = c('accuracy'))
# model %>% fit(train, train_label, epochs = 100, batch_size = 128, validation_split = 0.2, callbacks = c(callback_early_stopping(monitor = "val_accuracy", patience = 20, restore_best_weights = TRUE)))
# loss_and_metrics <- model %>% evaluate(test, test_label, batch_size = 128)