require(MASS)
library(class)
library(rpart)
library(caret)
library(MLmetrics)
library(haven)
library(mlr)
library(UBL)
library(ROSE)
library(rpart.plot)
library(dplyr)
library(party)
library(pROC)
library(lattice)
library(corrplot)
library(mlr)
library(sqldf)
library(ggplot2)
library(klaR)
library(pivottabler)
library(tidyverse)
# Para SVM y Naive Bayes
library(e1071)
library (plyr)
library(viridis)
library(hrbrthemes)
library(tictoc)
library(DataExplorer)
# Para XGBOOST
#install.packages("xgboost")
library("xgboost")
# Para Random Forest
library("randomForest")
# Funcion util

calcular_AUC <- function(y_pred, y_true){
  rank <- rank(y_pred)
  n_pos <- as.numeric(sum(y_true == 1))
  n_neg <- as.numeric(sum(y_true == 0))
  auc <- (sum(rank[y_true == 1]) - n_pos * (n_pos + 1)/2)/(n_pos * n_neg)
  return(auc)
}

consolidar_indicadores <- function(Algoritmo,target_y,target_pred,df_indicadores) {
  # Indicadores
  #AUC     <- MLmetrics::AUC(as.numeric(as.character(target_pred)),as.numeric(as.character(target_y)))
  AUC     <- calcular_AUC(as.numeric(as.character(target_pred)),as.numeric(as.character(target_y)))
  #GINI    <- 2*AUC-1
  GINI    <- MLmetrics::Gini(as.numeric(as.character(target_pred)),as.numeric(as.character(target_y)))
  KS      <- MLmetrics::KS_Stat(as.numeric(as.character(target_pred)),as.numeric(as.character(target_y)))
  LogLoss <- MLmetrics::LogLoss(as.numeric(as.character(target_pred)),as.numeric(as.character(target_y)))
  
  # Calcular los valores predichos
  #PRED <- as.factor(ifelse(target_pred <= mean(target_pred),0,1)) # Para regresion logistica
  #PRED <- as.factor(target_pred) # Si ya viene parseada la logica de las target binomial
  
  # Calcular la matriz de confusion
  #tabla <- caret::confusionMatrix(PRED,target_y,positive = "1") # 
  #tabla
  
  # sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=target_y,y_pred=target_pred)
  
  # especificidad
  Specificity <-  MLmetrics::Specificity(y_true=target_y,y_pred=target_pred) #Specificity(y_true, y_pred, positive = NULL)
  
  # Precision
  Accuracy <- MLmetrics::Accuracy(y_true=target_y,y_pred=target_pred)
  
  # F1-score
  f1_Score <- MLmetrics::F1_Score(y_true=target_y,y_pred=target_pred)
  # Agregar resultados al dataframe
  
  return( c(Algoritmo,AUC,GINI,KS,LogLoss,Accuracy,Sensitivity,Specificity,f1_Score))
}
##############
datos <- read.csv("data fec al 19-10-2022 original.csv",sep = ",")

# Obteniendo solo los campos pertinentes para el calculo
datos_filter <- datos[c(19,20,21,22,23,24,26,27,28,25,29,30,31,32,33,34,35,36)]

# Grafica de valores faltantes solo de las variables que ingresaran al modelo
plot_missing(datos_filter)

# Conversion de datos
datos_filter <- sqldf("
select
      case
      when ind_reclamo_regularizacion=='SI' then 1 else 0 end ind_reclamo_regularizacion,
      case
      when ind_retiro_regularizacion=='SI' then 1 else 0 end ind_retiro_regularizacion,
      case
      when ind_refinanciamiento=='SI' then 1 else 0 end ind_refinanciamiento,
      case
      when ind_aviso_ficticio=='SI' then 1 else 0 end ind_aviso_ficticio,
      case
      when ind_considerar_fec=='SI' then 1 else 0 end ind_considerar_fec,
      case
      when ind_limite_pago_fec=='SI' then 1 else 0 end ind_limite_pago_fec,
      case
      when ind_ffvv_vida=='SI' then 1 else 0 end ind_ffvv_vida,
      case
      when ind_cartera_huerfana=='SI' then 1 else 0 end ind_cartera_huerfana,
      case
      when ind_aviso_fec=='SI' then 1 else 0 end ind_aviso_fec,
      case
      when ind_bsc_fec=='SI' then 1 else 0 end ind_bsc_fec,
      mnt_prima_emitida_usd, 
      mnt_prima_cobrada_usd, 
      mnt_prima_emitida_usd_fec, 
      mnt_prima_cobrada_usd_fec, 
      num_cuota_pagada_ajustada_fec, 
      num_cuota_pagada_fec, 
      mnt_prima_emitida_usd_ajustada_fec,
      mnt_prima_cobrada_usd_ajustada_fec
from datos_filter")

datos_filter$ind_reclamo_regularizacion <- as.numeric(datos_filter$ind_reclamo_regularizacion)
datos_filter$ind_retiro_regularizacion <- as.numeric(datos_filter$ind_retiro_regularizacion)
datos_filter$ind_refinanciamiento <- as.numeric(datos_filter$ind_refinanciamiento)
datos_filter$ind_aviso_ficticio <- as.numeric(datos_filter$ind_aviso_ficticio)
datos_filter$ind_considerar_fec <- as.numeric(datos_filter$ind_considerar_fec)
datos_filter$ind_limite_pago_fec <- as.numeric(datos_filter$ind_limite_pago_fec)
datos_filter$ind_ffvv_vida <- as.numeric(datos_filter$ind_ffvv_vida)
datos_filter$ind_cartera_huerfana <- as.numeric(datos_filter$ind_cartera_huerfana)
datos_filter$ind_aviso_fec <- as.numeric(datos_filter$ind_aviso_fec)
datos_filter$ind_bsc_fec <- as.factor(datos_filter$ind_bsc_fec)

data.frame(table(datos_filter$ind_bsc_fec))

tabla_contingencia_target_completa <- data.frame(table(datos_filter$ind_bsc_fec))

data.frame(prop.table(table(datos_filter$ind_bsc_fec))) # Porcentaje de la target

############### CORRELACION ####################
## ojo: siempre y cuando aplique realizarlo
## ver correlacion antes de categorizar
source("funciones.R")
# SOLO PARA EVALUAR CORRELACION LA TARGET VOLVERA A SER NUMERIC

datos_filter$ind_bsc_fec <- as.numeric(as.character(datos_filter$ind_bsc_fec))
# Variables con peso sin correlacion
datos_filter_corre<- datos_filter[c(
                       "ind_bsc_fec" ,
                       "ind_reclamo_regularizacion" ,
                       "ind_aviso_ficticio" ,
                       "mnt_prima_cobrada_usd" ,
                       "ind_ffvv_vida",
                       "ind_limite_pago_fec")
                       ]
corre <- cor(datos_filter_corre,method = c("spearman"))

## colocamos la primera funcion de correlacion
corre <- correlacionS(corre)
corre$filtro <- ifelse(abs(corre$cor)>0.6,1,0)

write.csv(corre, file = "correlacion.csv")
datos_filter<- datos_filter_corre
########### PARTICION MUESTRAL #################  
set.seed(123)
training.samples <- datos_filter$ind_bsc_fec %>% 
  createDataPartition(p = 0.7, list = FALSE)
train  <- datos_filter[training.samples, ]
test <- datos_filter[-training.samples, ]


train$ind_bsc_fec <- as.factor(train$ind_bsc_fec)
test$ind_bsc_fec <- as.factor(test$ind_bsc_fec)

tabla_contingencia_target_train <- data.frame(table(train$ind_bsc_fec))

# Para RL
# Modelo
formula <-    ind_bsc_fec ~ 
              ind_reclamo_regularizacion +
              ind_retiro_regularizacion +
              ind_refinanciamiento +
              ind_aviso_ficticio +
              ind_considerar_fec +
              ind_limite_pago_fec +
              ind_ffvv_vida +
              ind_cartera_huerfana +
              ind_aviso_fec +
              mnt_prima_emitida_usd +
              mnt_prima_cobrada_usd +
              mnt_prima_emitida_usd_fec +
              mnt_prima_cobrada_usd_fec +
              num_cuota_pagada_ajustada_fec +
              num_cuota_pagada_fec +
              mnt_prima_emitida_usd_ajustada_fec +
              mnt_prima_cobrada_usd_ajustada_fec

modelo1 <- glm(formula,data=train,family = binomial(link = "logit"), 
               na.action = "na.omit", method = "glm.fit" )

summary(modelo1)

# Prediccion TEST
predicciones <- predict(modelo1, train, se.fit = TRUE)
# Mediante la función logit se transforman los log_ODDs a probabilidades.
predicciones_logit <- exp(predicciones$fit) / (1 + exp(predicciones$fit))

# Prediccion falsa para localizar el ideal con Youden
pred_logistico <- ifelse(predicciones_logit>0.50,1,0)

## Curvas ROC
# Area debajo de la curva ROC
analysis <- roc(response=train$ind_bsc_fec, predictor=pred_logistico)
analysis

# Grafica de la Curva ROC
plot(1-analysis$specificities,analysis$sensitivities,type="l",
     ylab="Sensitividad",xlab="1-Especificidad",col="black",lwd=2,
     main = "Curva ROC para el modelo logistico")
abline(a=0,b=1)

# Hallar punto de corte (para cualquier clasificador)
# Usando el criterio del índice J de Youden
# J = Sensitivity + Specificity - 1
e <- cbind(analysis$thresholds,analysis$sensitivities+analysis$specificities-1)
head(e)
pto_corte_jouden <- subset(e,e[,2]==max(e[,2]))[,2]#No deberia ser [,2]?
# Punto de corte segun youden del train
pto_corte_jouden

# Prediccion para el test con criterio de youden
predicciones <- predict(modelo1, test, se.fit = TRUE)
# Mediante la función logit se transforman los log_ODDs a probabilidades.
predicciones_logit <- exp(predicciones$fit) / (1 + exp(predicciones$fit))

# Prediccion falsa para localizar el ideal con Youden
pred_logistico_youden <- ifelse(predicciones_logit>pto_corte_jouden,1,0)

# UNIENDO LA PREDICCION AL DF
datos_prediccion<- cbind(test,pred_logistico_youden)


# Prediccion para el test con criterio de suma del promedio de probabilidades
prom_prob <- mean(predict(modelo1, type="response"))  # CRITERIO DEL PROMEDIO DE PROBABILIDADES DEL TRAIN
prom_prob
predicciones <- predict(modelo1, test, se.fit = TRUE)
# Mediante la función logit se transforman los log_ODDs a probabilidades.
predicciones_logit <- exp(predicciones$fit) / (1 + exp(predicciones$fit))

# Prediccion falsa para localizar el ideal con Youden
pred_logistico_crit_sum_prob <- ifelse(predicciones_logit>prom_prob,1,0)

# UNIENDO LA PREDICCION AL DF
datos_prediccion<- cbind(datos_prediccion,pred_logistico_crit_sum_prob)

## Para SVM (EXCLUIDO)
## Lineal
#modelo2 <- svm(formula = formula, data = train, kernel = "linear",
#               na.action = na.omit, scale = TRUE,
#               Type='C-classification')
#summary(modelo2)
## Prediccion TEST
#pred_svm_lineal <- predict(modelo2, test)
## UNIENDO EL RESULTADO AL DF
#datos_prediccion<- cbind(datos_prediccion,pred_svm_lineal)
#
## Poly
#modelo3 <- svm(formula = formula, data = train, kernel = "polynomial",
#               na.action = na.omit, #scale = TRUE,
#               Type='C-classification')
#summary(modelo3)
## Prediccion TEST
#pred_svm_poly <- predict(modelo3, test)
## UNIENDO EL RESULTADO AL DF
#datos_prediccion<- cbind(datos_prediccion,pred_svm_poly)
#
## Sigmoide
#modelo4 <- svm(formula = formula, data = train, kernel = "sigmoid",
#               na.action = na.omit, scale = FALSE,
#               Type='C-classification')
#summary(modelo4)
## Prediccion TEST
#pred_svm_sigmoide <- predict(modelo4, test)
## UNIENDO EL RESULTADO AL DF
#datos_prediccion<- cbind(datos_prediccion,pred_svm_sigmoide)

## PARA KNN  (EXCLUIDO)
## K SIMPLE
#knn_simple<-knn(train = train[,-10],test = test[,-10],cl = train$ind_bsc_fec)
##Estimacion del error por resubstitución
#confusionMatrix(knn_simple,test$ind_bsc_fec)
#datos_prediccion<- cbind(datos_prediccion,knn_simple)
#
## K = 3
#knn_3<-knn(train = train[,-10],test = test[,-10],cl = train$ind_bsc_fec, k = 3)
##Estimacion del error por resubstitución
#confusionMatrix(knn_3,test$ind_bsc_fec)
#datos_prediccion<- cbind(datos_prediccion,knn_3)
#
## K = 5
#knn_5<-knn(train = train[,-10],test = test[,-10],cl = train$ind_bsc_fec, k = 5)
##Estimacion del error por resubstitución
#confusionMatrix(knn_5,test$ind_bsc_fec)
#datos_prediccion<- cbind(datos_prediccion,knn_5)
#
## K = 7
#knn_7<-knn(train = train[,-10],test = test[,-10],cl = train$ind_bsc_fec, k = 7)
##Estimacion del error por resubstitución
#confusionMatrix(knn_7,test$ind_bsc_fec)
#datos_prediccion<- cbind(datos_prediccion,knn_7)
#
## K = 15
#knn_15<-knn(train = train[,-10],test = test[,-10],cl = train$ind_bsc_fec, k = 15)
##Estimacion del error por resubstitución
#confusionMatrix(knn_15,test$ind_bsc_fec)
#datos_prediccion<- cbind(datos_prediccion,knn_15)

# PARA ARBOL DE DESICION CART 
# Pagina de referencia de los parametros: https://cran.r-project.org/web/packages/rpart/rpart.pdf

arbol.completo <- rpart::rpart(formula,data = train, method="class",cp=0, minbucket=0)

rpart.plot::rpart.plot(arbol.completo, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

xerr    <- arbol.completo$cptable[,"xerror"] ## error de la validacion cruzada
minxerr <- which.min(xerr)
mincp   <- arbol.completo$cptable[minxerr, "CP"]

# Poda del arbol
modelo2 <- rpart::prune(arbol.completo,cp=mincp)

rpart.plot::rpart.plot(modelo2, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)
# Prediccion TEST
pred_arbol_cart <- predict(modelo2, test,type="class")
# UNIENDO EL RESULTADO AL DF
datos_prediccion<- cbind(datos_prediccion,pred_arbol_cart)

# PARA NAIVE BAYES
modelo5 <- naiveBayes(formula, data = train, na.action = na.omit)
summary(modelo5)
# Prediccion TEST
pred_naive_bayes <- predict(modelo5, test)
# UNIENDO EL RESULTADO AL DF
datos_prediccion<- cbind(datos_prediccion,pred_naive_bayes)

# PARA XGBOOST

# Para formula completa
modelo6 <- xgboost(data = as.matrix(train[,-10]),label = as.matrix(train$ind_bsc_fec), 
        objective = "binary:logistic",
        nrounds = 10, max.depth = 2, eta = 0.3, nthread = 2)

summary(modelo6)
# Prediccion TEST
pred_xgboost <- predict(modelo6, as.matrix(test[,-10]))
pred_xgboost <- ifelse(pred_xgboost>pto_corte_jouden,1,0)
# UNIENDO EL RESULTADO AL DF
datos_prediccion<- cbind(datos_prediccion,pred_xgboost)
datos_prediccion$ind_bsc_fec<- as.numeric(as.character(datos_prediccion$ind_bsc_fec))

# Para formula optimizada

modelo6 <- xgboost(data = as.matrix(train[,-1]),label = as.matrix(train$ind_bsc_fec), 
                   objective = "binary:logistic",
                   nrounds = 10, max.depth = 2, eta = 0.3, nthread = 2)

summary(modelo6)
# Prediccion TEST
pred_xgboost <- predict(modelo6, as.matrix(test[,-1]))
pred_xgboost <- ifelse(pred_xgboost>pto_corte_jouden,1,0)
# UNIENDO EL RESULTADO AL DF
datos_prediccion<- cbind(datos_prediccion,pred_xgboost)
datos_prediccion$ind_bsc_fec<- as.numeric(as.character(datos_prediccion$ind_bsc_fec))


# Creando la tabla de indicadores vacia
df_indicadores <- data.frame(Algoritmo = character(),
                             AUC = character(),
                             GINI = character(),
                             KS = character(),
                             LogLoss = character(),
                             Accuracy = character(),
                             Sensibilidad = character(),
                             Especificidad= character(),
                             F1_score= character())

######### Calculando los indicadores del modelo ###########
df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('Logistico - Indice Youden',datos_prediccion$ind_bsc_fec,
                                                                  datos_prediccion$pred_logistico_youden,df_indicadores)
df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('Logistico - Criterio de suma de probabilidades',datos_prediccion$ind_bsc_fec,
                                                                  datos_prediccion$pred_logistico_crit_sum_prob,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('SVM Lineal',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$pred_svm_lineal,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('SVM Poly',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$pred_svm_poly,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('SVM Sigmoide',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$pred_svm_sigmoide,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('KNN Simple',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$knn_simple,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('KNN K = 3',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$knn_3,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('KNN K = 5',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$knn_5,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('KNN K = 7',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$knn_7,df_indicadores)
#df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('KNN K = 15',datos_prediccion$ind_bsc_fec,
#                                                                  datos_prediccion$knn_15,df_indicadores)
df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('Arbol CART',datos_prediccion$ind_bsc_fec,
                                                                  datos_prediccion$pred_arbol_cart,df_indicadores)
df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('Naive Bayes',datos_prediccion$ind_bsc_fec,
                                                                  datos_prediccion$pred_naive_bayes,df_indicadores)
df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores('XGBOOST',datos_prediccion$ind_bsc_fec,
                                                                  datos_prediccion$pred_xgboost,df_indicadores)

write.csv(df_indicadores, file = "df_indicadores reg logistica.csv")


######### CROSS VALIDATION ###########
# CALCULO DEL TIEMPO
# Regresion Logistica - con criterio youden
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvRegresionLogisticaYouden <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  clasificador <- glm(formula, family = binomial, data = training_fold)
  y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- ifelse(y_pred > pto_corte_jouden, 1, 0)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadRegresionLogisticaYouden <- mean(as.numeric(cvRegresionLogisticaYouden))
sensibilidadRegresionLogisticaYouden

# Parseando a tabla el resultado
df_cv <- NULL
Algoritmo <- 'Logistico con criterio Youden'
Metrica <- 'Sensibilidad'
df_cv <- cbind(Algoritmo,Metrica, ldply (cvRegresionLogisticaYouden, data.frame) )
df_cv
# Tiempo
df_tiempo <- NULL
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- cbind(Algoritmo,tiempo)


# Regresion Logistica - con suma probabilidades
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvRegresionLogisticaSumProb <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  clasificador <- glm(formula, family = binomial, data = training_fold)
  youden_cv_temp <- mean(predict(clasificador, type="response"))
  youden_cv_temp
  y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- ifelse(y_pred > youden_cv_temp, 1, 0)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadRegresionLogisticaSumProb <- mean(as.numeric(cvRegresionLogisticaSumProb))
sensibilidadRegresionLogisticaSumProb
Algoritmo <- 'Regresion Logistica con suma probabilidades'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvRegresionLogisticaSumProb, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))




# KNN SIMPLE
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvknnSimple <- lapply(folds, function(x){
 training_fold <- train[-x, ]
 test_fold <- train[x, ]
 #clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
 #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
 y_pred <- knn(train = training_fold[,-10],test = test_fold[,-10],cl = training_fold$ind_bsc_fec)
 #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
 #cm <- table(test_fold$TipoEstrella, y_pred)
 #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
 # Sensibilidad
 Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
 return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadKNNSimple <- mean(as.numeric(cvknnSimple))
sensibilidadKNNSimple
Algoritmo <- 'KNN Simple'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvknnSimple, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))

# KNN K = 3
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvknn3 <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  #clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
  #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- knn(train = training_fold[,-10],test = test_fold[,-10],cl = training_fold$ind_bsc_fec)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadKNN3 <- mean(as.numeric(cvknn3))
sensibilidadKNN3
Algoritmo <- 'KNN K = 3'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvknn3, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# KNN K = 5
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvknn5 <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  #clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
  #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- knn(train = training_fold[,-10],test = test_fold[,-10],cl = training_fold$ind_bsc_fec)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadKNN5 <- mean(as.numeric(cvknn5))
sensibilidadKNN5
Algoritmo <- 'KNN K = 5'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvknn5, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# KNN K = 7
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvknn7 <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  #clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
  #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- knn(train = training_fold[,-10],test = test_fold[,-10],cl = training_fold$ind_bsc_fec)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadKNN7 <- mean(as.numeric(cvknn7))
sensibilidadKNN7
Algoritmo <- 'KNN K = 7'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvknn7, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# KNN K = 15
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
cvknn15 <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  #clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
  #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- knn(train = training_fold[,-10],test = test_fold[,-10],cl = training_fold$ind_bsc_fec)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadKNN15 <- mean(as.numeric(cvknn15))
sensibilidadKNN15
Algoritmo <- 'KNN K = 15'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvknn15, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# Naiva Bayes
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
naiveBayes <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  clasificador <- naiveBayes(formula = formula, data = training_fold, na.action = na.omit)
  #y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
  y_pred <- predict(clasificador, test_fold)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadNaiveBayes <- mean(as.numeric(naiveBayes))
sensibilidadNaiveBayes
Algoritmo <- 'Naive Bayes'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (naiveBayes, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# XGBOOST
startTime <- Sys.time()
folds <- createFolds(train$ind_bsc_fec, k = 10)
xgboost <- lapply(folds, function(x){
  training_fold <- train[-x, ]
  test_fold <- train[x, ]
  clasificador <- xgboost(data = as.matrix(training_fold[,-10]),label = as.matrix(training_fold$ind_bsc_fec), 
                          objective = "binary:logistic",
                          nrounds = 10, max.depth = 2, eta = 0.3, nthread = 2)
  # Prediccion TEST
  y_pred <- predict(clasificador, as.matrix(test_fold[,-10]))
  y_pred <- ifelse(y_pred>pto_corte_jouden,1,0)
  #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
  #cm <- table(test_fold$TipoEstrella, y_pred)
  #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
  # Sensibilidad
  Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
  return(Sensitivity)
})
endTime <- Sys.time()
sensibilidadXGBoost <- mean(as.numeric(xgboost))
sensibilidadXGBoost
Algoritmo <- 'XGBoost'
Metrica <- 'Sensibilidad'
df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (xgboost, data.frame)) )
# Tiempo
tiempo <- as.numeric(endTime - startTime)
df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


# # SVM LINEAL
# startTime <- Sys.time()
# folds <- createFolds(train$ind_bsc_fec, k = 10)
# cvSVMLineal <- lapply(folds, function(x){
#   training_fold <- train[-x, ]
#   test_fold <- train[x, ]
#   clasificador <- svm(formula = formula, data = training_fold, kernel = "linear", na.action = na.omit, scale = TRUE, Type='C-classification')
#   y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
#   #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
#   #cm <- table(test_fold$TipoEstrella, y_pred)
#   #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
#   # Sensibilidad
#   Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
#   return(Sensitivity)
# })
# endTime <- Sys.time()
# sensibilidadSVMLineal <- mean(as.numeric(cvSVMLineal))
# sensibilidadSVMLineal
# Algoritmo <- 'SVM Lineal'
# Metrica <- 'Sensibilidad'
# df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvSVMLineal, data.frame)) )
# # Tiempo
# tiempo <- as.numeric(endTime - startTime)
# df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))
# 
# 
# # SVM POLINOMIAL
# startTime <- Sys.time()
# folds <- createFolds(train$ind_bsc_fec, k = 10)
# cvSVMPolinomial <- lapply(folds, function(x){
#   training_fold <- train[-x, ]
#   test_fold <- train[x, ]
#   clasificador <- svm(formula = formula, data = training_fold, kernel = "polynomial", na.action = na.omit, scale = TRUE, Type='C-classification')
#   y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
#   #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
#   #cm <- table(test_fold$TipoEstrella, y_pred)
#   #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
#   # Sensibilidad
#   Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
#   return(Sensitivity)
# })
# endTime <- Sys.time()
# sensibilidadSVMPolinomial <- mean(as.numeric(cvSVMPolinomial))
# sensibilidadSVMPolinomial
# Algoritmo <- 'SVM Polinomial'
# Metrica <- 'Sensibilidad'
# df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvSVMPolinomial, data.frame)) )
# # Tiempo
# tiempo <- as.numeric(endTime - startTime)
# df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))
# 
# 
# # SVM SIGMOIDE
# startTime <- Sys.time()
# folds <- createFolds(train$ind_bsc_fec, k = 10)
# cvSVMSigmoide <- lapply(folds, function(x){
#   training_fold <- train[-x, ]
#   test_fold <- train[x, ]
#   clasificador <- svm(formula = formula, data = training_fold, kernel = "sigmoid", na.action = na.omit, scale = TRUE, Type='C-classification')
#   y_pred <- predict(clasificador, type = 'response', newdata = test_fold)
#   #y_pred <- factor(y_pred, levels = c("0", "1"), labels = c("NoPulsar", "Pulsar"))
#   #cm <- table(test_fold$TipoEstrella, y_pred)
#   #precision <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] +cm[1,2] + cm[2,1])
#   # Sensibilidad
#   Sensitivity <-  MLmetrics::Sensitivity(y_true=test_fold$ind_bsc_fec,y_pred=y_pred)
#   return(Sensitivity)
# })
# endTime <- Sys.time()
# sensibilidadSVMSigmoide <- mean(as.numeric(cvSVMSigmoide))
# sensibilidadSVMSigmoide
# Algoritmo <- 'SVM Sigmoide'
# Metrica <- 'Sensibilidad'
# df_cv <- rbind( df_cv, cbind( Algoritmo , Metrica,  ldply (cvSVMSigmoide, data.frame)) )
# # Tiempo
# tiempo <- as.numeric(endTime - startTime)
# df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))
# 


# Grafica de validacion CROSS VALIDATION
#obj <- df_cv %>% 
#  filter(Algoritmo %in% c("Logistico", "SVM Lineal"))
# Plot
df_cv %>%
  ggplot( aes(x= .id, y= X..i.., group=Algoritmo, color=Algoritmo)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Sensibilidad en Cross Validation por algoritmo") +
  theme_ipsum() +
  ylab("Sensibilidad")+xlab("K - iteraciones")+ 
  geom_point(size=1) + 
  geom_label(label = round(df_cv$X..i.. , 3))


# Grafica de tiempo de ejecucion
df_tiempo <- as.data.frame(df_tiempo)
df_tiempo$tiempo <- round(as.numeric(df_tiempo$tiempo),3)

df_tiempo  %>%
ggplot( aes(x=Algoritmo, y=tiempo, fill=Algoritmo)) +
  geom_bar(stat="identity") +
  xlab("Tiempo en segundos")+
  geom_col() +
  geom_text(aes(y = tiempo, label = tiempo), colour = "white") 

# FIN DE CODIGO

# OPCIONAL
# GRID SEARCH
# RUTA DE REFERENCIA LIBRERIA: https://topepo.github.io/caret/available-models.html
# Para Regresion Logistica
clasificadorTuning <- caret::train( form = formula, data = train, method = "glm")
clasificadorTuning

# Para SVM
# Lineal
clasificadorTuning <- caret::train( form = formula, data = train, method = "svmLinear2")
clasificadorTuning

## alternatively soporte para tunes da el parametro gamma y cost para la libreria en general e1071
obj <- tune.svm(formula, data = train, gamma = 2^(-1:1), cost = 2^(2:4) )
summary(obj)
plot(obj)


# Polynomial # No es la misma libreria utilizada
clasificadorTuning <- caret::train( form = formula, data = train, method = "svmPoly")
clasificadorTuning

# Sigmoide # no hay soporte para esta categoría en caret
#clasificadorTuning <- caret::train( form = formula, data = train, method = "svmPoly")
#clasificadorTuning





###################################################################################################
################## ITERACION DE PARAMETROS - ALGORITMOS ###########################################
###################################################################################################
# Creando la tabla de indicadores vacia
df_indicadores <- data.frame(Algoritmo = character(),
                             AUC = character(),
                             GINI = character(),
                             KS = character(),
                             LogLoss = character(),
                             Accuracy = character(),
                             Sensibilidad = character(),
                             Especificidad= character(),
                             F1_score= character())

df_tiempo <- data.frame(algoritmo = character(),
                             inicio = character(),
                             fin = character(),
                             duracion = character()
              )

# Ejecutar desde la linea formula 130
# NAIVE BAYES
df_multi_parameters_algortimos <- 
      data.frame(
        iteracion = c('Naive Bayes 1', 'Naive Bayes 2', 'Naive Bayes 3', 'Naive Bayes 4', 'Naive Bayes 5', 'Naive Bayes 6', 'Naive Bayes 7', 'Naive Bayes 8', 'Naive Bayes 9', 'Naive Bayes 10', 'Naive Bayes 11', 'Naive Bayes 12', 'Naive Bayes 13', 'Naive Bayes 14', 'Naive Bayes 15', 'Naive Bayes 16', 'Naive Bayes 17', 'Naive Bayes 18', 'Naive Bayes 19', 'Naive Bayes 20', 'Naive Bayes 21', 'Naive Bayes 22'),
        laplace = c(
          '0','1','2','3','4','5','6','7','8','9','10','0','1','2','3','4','5','6','7','8','9','10'
        ),
        
        threshold = c('0.001','0.001','0.001','0.001','0.001','0.001','0.001','0.001','0.001','0.001','0.001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001','0.0001'
        ),
        
        eps = c('0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'
        )
      )
# Formula cortada
formula <-    ind_bsc_fec ~ 
  ind_reclamo_regularizacion +
  ind_aviso_ficticio +
  mnt_prima_cobrada_usd +
  ind_ffvv_vida+
  ind_limite_pago_fec
  
# Parseando a tabla el resultado
#for (i in 1:22) {
for (i in 1:22) {
  startTime <- Sys.time()
  Algoritmo <- df_multi_parameters_algortimos[i,1]
  # Creamos el data frame datos_prediccion
  datos_prediccion<- as.data.frame(cbind(test))
  datos_prediccion$ind_bsc_fec <- as.numeric(as.character(datos_prediccion$ind_bsc_fec ))
  
  #PARAMETROS
  print(df_multi_parameters_algortimos[i,1])
  # PARA NAIVE BAYES
  modelo5 <- naiveBayes(formula, data = train, na.action = na.omit,laplace= df_multi_parameters_algortimos[i,2],
                        threshold= df_multi_parameters_algortimos[i,3], eps= df_multi_parameters_algortimos[i,4] )
  #summary(modelo5)
  # Prediccion TEST
  pred_naive_bayes <- as.numeric(as.character(predict(modelo5, test)))
  # UNIENDO EL RESULTADO AL DF
  datos_prediccion<- as.data.frame(cbind(datos_prediccion,pred_naive_bayes))
  # CALCULO DE INDICADORES
  df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores(df_multi_parameters_algortimos[i,1],
                                      datos_prediccion$ind_bsc_fec, datos_prediccion$pred_naive_bayes,df_indicadores)
  endTime <- Sys.time()
  tiempo <- as.numeric(endTime - startTime)
  df_tiempo[nrow(df_tiempo) + 1,] = c(Algoritmo,as.character(startTime),as.character(endTime),as.character(tiempo))
  
}
# Tiempo

#df_tiempo <- cbind(Algoritmo,tiempo)

# df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))


summary(datos_prediccion)
write.csv(df_indicadores, file = "metricas naive bayes 1.csv")
write.csv(df_tiempo, file = "df tiempo naive bayes 1.csv")
write.csv(df_indicadores, file = "metricas naive bayes 2.csv")
write.csv(df_tiempo, file = "df tiempo naive bayes 2.csv")
print(df_multi_parameters_algortimos[1,1])











# XGBOOST
# Primera iteracion de parametros
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3),
  nthread= c(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
)

# Segunda iteracion de parametros
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30','XGBOOST 31','XGBOOST 32','XGBOOST 33','XGBOOST 34','XGBOOST 35','XGBOOST 36','XGBOOST 37','XGBOOST 38','XGBOOST 39','XGBOOST 40','XGBOOST 41','XGBOOST 42','XGBOOST 43','XGBOOST 44','XGBOOST 45','XGBOOST 46','XGBOOST 47','XGBOOST 48','XGBOOST 49','XGBOOST 50','XGBOOST 51','XGBOOST 52','XGBOOST 53','XGBOOST 54','XGBOOST 55','XGBOOST 56','XGBOOST 57','XGBOOST 58','XGBOOST 59','XGBOOST 60','XGBOOST 61','XGBOOST 62','XGBOOST 63','XGBOOST 64','XGBOOST 65','XGBOOST 66','XGBOOST 67','XGBOOST 68','XGBOOST 69','XGBOOST 70','XGBOOST 71','XGBOOST 72','XGBOOST 73','XGBOOST 74','XGBOOST 75','XGBOOST 76','XGBOOST 77','XGBOOST 78','XGBOOST 79','XGBOOST 80'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01),
  nthread= c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
)

# Tercera iteracion de parametros
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4),
  nthread= c(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)
# Cuarta iteracion de parametros con formula completa
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5),
  nthread= c(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)
# Quinta iteracion de parametros con formula completa
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6),
  nthread= c(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)
# Sexta iteracion de parametros con formula completa
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('XGBOOST 1','XGBOOST 2','XGBOOST 3','XGBOOST 4','XGBOOST 5','XGBOOST 6','XGBOOST 7','XGBOOST 8','XGBOOST 9','XGBOOST 10','XGBOOST 11','XGBOOST 12','XGBOOST 13','XGBOOST 14','XGBOOST 15','XGBOOST 16','XGBOOST 17','XGBOOST 18','XGBOOST 19','XGBOOST 20','XGBOOST 21','XGBOOST 22','XGBOOST 23','XGBOOST 24','XGBOOST 25','XGBOOST 26','XGBOOST 27','XGBOOST 28','XGBOOST 29','XGBOOST 30'),
  nrounds= c(10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1),
  max_depth = c(2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4),
  eta= c(0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7),
  nthread= c(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)
# Formula cortada
formula <-    ind_bsc_fec ~ 
  ind_reclamo_regularizacion +
  ind_aviso_ficticio +
  mnt_prima_cobrada_usd +
  ind_ffvv_vida+
  ind_limite_pago_fec

#for (i in 1:30)
for (i in 1:30) {
  startTime <- Sys.time()
  Algoritmo <- df_multi_parameters_algortimos[i,1]
  modelo6 <- xgboost(data = as.matrix(train[,-10]),label = as.matrix(train$ind_bsc_fec), 
                     objective = "binary:logistic",
                     nrounds = df_multi_parameters_algortimos[i,2], max.depth = df_multi_parameters_algortimos[i,3], 
                     eta = df_multi_parameters_algortimos[i,4],
                     nthread = df_multi_parameters_algortimos[i,5])
  #summary(modelo6)
  # Prediccion TEST
  pred_xgboost <- predict(modelo6, as.matrix(test[,-10]))
  pred_xgboost <- ifelse(pred_xgboost>pto_corte_jouden,1,0)
  
  # Creamos el data frame datos_prediccion
  datos_prediccion<- as.data.frame(cbind(test))
  datos_prediccion$ind_bsc_fec <- as.numeric(as.character(datos_prediccion$ind_bsc_fec ))
  
  # UNIENDO EL RESULTADO AL DF
  datos_prediccion<- cbind(datos_prediccion,pred_xgboost)
  
  # CALCULO DE INDICADORES
  df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores(df_multi_parameters_algortimos[i,1],
                                datos_prediccion$ind_bsc_fec, datos_prediccion$pred_xgboost,df_indicadores)
  
  endTime <- Sys.time()
  tiempo <- as.numeric(endTime - startTime)
  df_tiempo[nrow(df_tiempo) + 1,] = c(Algoritmo,as.character(startTime),as.character(endTime),as.character(tiempo))
}

#df_tiempo <- rbind(df_tiempo, cbind(Algoritmo,tiempo))
write.csv(df_indicadores, file = "metricas xgboost.csv")
write.csv(df_indicadores, file = "metricas xgboost 2.csv")
write.csv(df_indicadores, file = "metricas xgboost 3.csv")
write.csv(df_indicadores, file = "metricas xgboost 4.csv")
write.csv(df_indicadores, file = "metricas xgboost 5.csv")
write.csv(df_indicadores, file = "metricas xgboost 6.csv")
write.csv(df_indicadores, file = "metricas xgboost 7.csv")
write.csv(df_indicadores, file = "metricas xgboost 8.csv")
write.csv(df_indicadores, file = "metricas xgboost 9.csv")
write.csv(df_indicadores, file = "metricas xgboost 10.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 3.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 4.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 5.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 6.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 7.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 8.csv")
write.csv(df_tiempo, file = "df tiempo xgboost completo 10.csv")


# ARBOL DE DESICION CART
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Arbol CART 1','Arbol CART 2','Arbol CART 3','Arbol CART 4','Arbol CART 5','Arbol CART 6','Arbol CART 7','Arbol CART 8','Arbol CART 9','Arbol CART 10','Arbol CART 11','Arbol CART 12','Arbol CART 13','Arbol CART 14','Arbol CART 15','Arbol CART 16','Arbol CART 17','Arbol CART 18','Arbol CART 19','Arbol CART 20','Arbol CART 21','Arbol CART 22','Arbol CART 23','Arbol CART 24','Arbol CART 25','Arbol CART 26','Arbol CART 27','Arbol CART 28','Arbol CART 29','Arbol CART 30','Arbol CART 31','Arbol CART 32','Arbol CART 33','Arbol CART 34','Arbol CART 35','Arbol CART 36','Arbol CART 37','Arbol CART 38','Arbol CART 39','Arbol CART 40','Arbol CART 41','Arbol CART 42','Arbol CART 43','Arbol CART 44','Arbol CART 45'),
  minsplit = c(20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160),
  cp = c(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01),
  maxdepth = c(30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10),
  minbucket = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
)

# Luego probar con sin el minbucket
# SEGUNDA ITERACION CON PARAMETROS DIFERENTES
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Arbol CART 1','Arbol CART 2','Arbol CART 3','Arbol CART 4','Arbol CART 5','Arbol CART 6','Arbol CART 7','Arbol CART 8','Arbol CART 9','Arbol CART 10','Arbol CART 11','Arbol CART 12','Arbol CART 13','Arbol CART 14','Arbol CART 15','Arbol CART 16','Arbol CART 17','Arbol CART 18','Arbol CART 19','Arbol CART 20','Arbol CART 21','Arbol CART 22','Arbol CART 23','Arbol CART 24','Arbol CART 25','Arbol CART 26','Arbol CART 27','Arbol CART 28','Arbol CART 29','Arbol CART 30','Arbol CART 31','Arbol CART 32','Arbol CART 33','Arbol CART 34','Arbol CART 35','Arbol CART 36','Arbol CART 37','Arbol CART 38','Arbol CART 39','Arbol CART 40','Arbol CART 41','Arbol CART 42','Arbol CART 43','Arbol CART 44','Arbol CART 45'),
  minsplit = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
  cp = c(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01),
  maxdepth = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3),
  minbucket = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
)

# TERCERA ITERACION CON PARAMETROS DIFERENTES SIN MINBUCKET
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Arbol CART 1','Arbol CART 2','Arbol CART 3','Arbol CART 4','Arbol CART 5','Arbol CART 6','Arbol CART 7','Arbol CART 8','Arbol CART 9','Arbol CART 10','Arbol CART 11','Arbol CART 12','Arbol CART 13','Arbol CART 14','Arbol CART 15','Arbol CART 16','Arbol CART 17','Arbol CART 18','Arbol CART 19','Arbol CART 20','Arbol CART 21','Arbol CART 22','Arbol CART 23','Arbol CART 24','Arbol CART 25','Arbol CART 26','Arbol CART 27','Arbol CART 28','Arbol CART 29','Arbol CART 30','Arbol CART 31','Arbol CART 32','Arbol CART 33','Arbol CART 34','Arbol CART 35','Arbol CART 36','Arbol CART 37','Arbol CART 38','Arbol CART 39','Arbol CART 40','Arbol CART 41','Arbol CART 42','Arbol CART 43','Arbol CART 44','Arbol CART 45'),
  minsplit = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
  cp = c(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01),
  maxdepth = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)

# CUARTA ITERACION CON PARAMETROS DIFERENTES SIN MINBUCKET
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Arbol CART 1','Arbol CART 2','Arbol CART 3','Arbol CART 4','Arbol CART 5','Arbol CART 6','Arbol CART 7','Arbol CART 8','Arbol CART 9','Arbol CART 10','Arbol CART 11','Arbol CART 12','Arbol CART 13','Arbol CART 14','Arbol CART 15','Arbol CART 16','Arbol CART 17','Arbol CART 18','Arbol CART 19','Arbol CART 20','Arbol CART 21','Arbol CART 22','Arbol CART 23','Arbol CART 24','Arbol CART 25','Arbol CART 26','Arbol CART 27','Arbol CART 28','Arbol CART 29','Arbol CART 30','Arbol CART 31','Arbol CART 32','Arbol CART 33','Arbol CART 34','Arbol CART 35','Arbol CART 36','Arbol CART 37','Arbol CART 38','Arbol CART 39','Arbol CART 40','Arbol CART 41','Arbol CART 42','Arbol CART 43','Arbol CART 44','Arbol CART 45'),
  minsplit = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
  cp = c(0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005),
  maxdepth = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)

# QUINTA ITERACION CON PARAMETROS DIFERENTES SIN MINBUCKET
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Arbol CART 1','Arbol CART 2','Arbol CART 3','Arbol CART 4','Arbol CART 5','Arbol CART 6','Arbol CART 7','Arbol CART 8','Arbol CART 9','Arbol CART 10','Arbol CART 11','Arbol CART 12','Arbol CART 13','Arbol CART 14','Arbol CART 15','Arbol CART 16','Arbol CART 17','Arbol CART 18','Arbol CART 19','Arbol CART 20','Arbol CART 21','Arbol CART 22','Arbol CART 23','Arbol CART 24','Arbol CART 25','Arbol CART 26','Arbol CART 27','Arbol CART 28','Arbol CART 29','Arbol CART 30','Arbol CART 31','Arbol CART 32','Arbol CART 33','Arbol CART 34','Arbol CART 35','Arbol CART 36','Arbol CART 37','Arbol CART 38','Arbol CART 39','Arbol CART 40','Arbol CART 41','Arbol CART 42','Arbol CART 43','Arbol CART 44','Arbol CART 45'),
  minsplit = c(10,11,12,13,14,15,16,17,18,10,11,12,13,14,15,16,17,18,10,11,12,13,14,15,16,17,18,10,11,12,13,14,15,16,17,18,10,11,12,13,14,15,16,17,18),
  cp = c(0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005),
  maxdepth = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
)

for (i in 1:2) {
  print(df_multi_parameters_algortimos[i,1])
  print(df_multi_parameters_algortimos[i,2])
  print(df_multi_parameters_algortimos[i,3])
  print(df_multi_parameters_algortimos[i,4])
}
#for (i in 1:45)
for (i in 1:45) {
  startTime <- Sys.time()
  Algoritmo <- df_multi_parameters_algortimos[i,1]
  arbol.completo <- rpart::rpart(formula,data = train, method="class",
                                 minsplit=df_multi_parameters_algortimos[i,2],
                                 cp=df_multi_parameters_algortimos[i,3], 
                                 maxdepth = df_multi_parameters_algortimos[i,4]#,
                                # minbucket= df_multi_parameters_algortimos[i,5]
                                 )
  #rpart.plot::rpart.plot(arbol.completo, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)
  xerr    <- arbol.completo$cptable[,"xerror"] ## error de la validacion cruzada
  minxerr <- which.min(xerr)
  mincp   <- arbol.completo$cptable[minxerr, "CP"]
  
  # Poda del arbol
  modelo2 <- rpart::prune(arbol.completo,cp=mincp)
  #modelo2 <- arbol.completo
  #rpart.plot::rpart.plot(modelo2, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)
  # Prediccion TEST
  pred_arbol_cart <- predict(modelo2, test,type="class")
  
  
  # Creamos el data frame datos_prediccion
  datos_prediccion<- as.data.frame(cbind(test))
  datos_prediccion$ind_bsc_fec <- as.numeric(as.character(datos_prediccion$ind_bsc_fec ))
  
  # UNIENDO EL RESULTADO AL DF
  datos_prediccion<- cbind(datos_prediccion,pred_arbol_cart)
  
  # CALCULO DE INDICADORES
  df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores(df_multi_parameters_algortimos[i,1],
                                          datos_prediccion$ind_bsc_fec, datos_prediccion$pred_arbol_cart,df_indicadores)
  
  endTime <- Sys.time()
  tiempo <- as.numeric(endTime - startTime)
  df_tiempo[nrow(df_tiempo) + 1,] = c(Algoritmo,as.character(startTime),as.character(endTime),as.character(tiempo))
  
}


write.csv(df_indicadores, file = "metricas arbol cart 1.csv")
write.csv(df_tiempo, file = "df tiempo arbol cart 1.csv")

write.csv(df_indicadores, file = "metricas arbol cart 4.csv")
write.csv(df_tiempo, file = "df tiempo arbol cart 4.csv")

write.csv(df_indicadores, file = "metricas arbol cart 5.csv")
write.csv(df_tiempo, file = "df tiempo arbol cart 5.csv")

# RANDOM FOREST
df_multi_parameters_algortimos <- data.frame(
  iteracion = c('Random Forest 1','Random Forest 2','Random Forest 3','Random Forest 4','Random Forest 5','Random Forest 6','Random Forest 7','Random Forest 8','Random Forest 9','Random Forest 10'),
  ntree = c(50,100,150,200,250,50,100,150,200,250),
  importance = c(TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE),
  mtry = c(3,3,3,3,3,4,4,4,4,4),
  proximity = c(TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE)
)
# Formula cortada
formula <-    ind_bsc_fec ~ 
  ind_reclamo_regularizacion +
  ind_aviso_ficticio +
  mnt_prima_cobrada_usd +
  ind_ffvv_vida+
  ind_limite_pago_fec

# Formula cortada
formula <-    ind_bsc_fec ~ 
  ind_reclamo_regularizacion+
  ind_aviso_ficticio+
  mnt_prima_cobrada_usd


data_prueba_for_randomforest<- train[,c('ind_bsc_fec','ind_reclamo_regularizacion',
                                        'ind_aviso_ficticio','mnt_prima_cobrada_usd')] 

modelo_randomForest <- randomForest(formula= formula,data = data_prueba_for_randomforest,
                                    method="classification",
                                    ntree=5,
                                    importance=TRUE, 
                                    mtry = 1,
                                    proximity = TRUE)

#for (i in 1:45)
for (i in 1:10) {
  startTime <- Sys.time()
  Algoritmo <- df_multi_parameters_algortimos[i,1]
  modelo_randomForest <- randomForest(formula,data = train, method="class",
                                      ntree=df_multi_parameters_algortimos[i,2],
                                      importance=df_multi_parameters_algortimos[i,3], 
                                      mtry = df_multi_parameters_algortimos[i,4],
                                      proximity = df_multi_parameters_algortimos[i,5]
  )
  # Prediccion TEST
  pred_randomForest <- predict(modelo_randomForest, test,type="class")
  
  
  # Creamos el data frame datos_prediccion
  datos_prediccion<- as.data.frame(cbind(test))
  datos_prediccion$ind_bsc_fec <- as.numeric(as.character(datos_prediccion$ind_bsc_fec ))
  
  # UNIENDO EL RESULTADO AL DF
  datos_prediccion<- cbind(datos_prediccion,pred_randomForest)
  
  # CALCULO DE INDICADORES
  df_indicadores[nrow(df_indicadores) + 1,] =consolidar_indicadores(df_multi_parameters_algortimos[i,1],
                                                                    datos_prediccion$ind_bsc_fec, datos_prediccion$pred_randomForest,df_indicadores)
  
  endTime <- Sys.time()
  tiempo <- as.numeric(endTime - startTime)
  df_tiempo[nrow(df_tiempo) + 1,] = c(Algoritmo,as.character(startTime),as.character(endTime),as.character(tiempo))
  
}

write.csv(df_indicadores, file = "metricas random forest.csv")
write.csv(df_tiempo, file = "df tiempo arbol random forest.csv")



###################################################################################
########################EXPLICABILIDAD CON SHAPR###################################
###################################################################################
library(shapr)

featureCombination <- feature_combinations(m = 17,exact = FALSE)
# Prepare the data for explanation
explainerobj <- shapr(as.matrix(train[,-10]), modelo6, n_combinations = 50) # Error por un numero grande de variables
# The specified model provides feature classes that are NA. The classes of data are taken as the truth.
# Error in feature_combinations(m = explainer$n_features, exact = explainer$exact,  : 
# Due to computational complexity, we recommend setting n_combinations = 10 000
# if the number of features is larger than 13. Note that you can force the use of the exact
# method (i.e. n_combinations = NULL) by setting n_combinations equal to 2^m,
# where m is the number of features.
#> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

# Specifying the phi_0, i.e. the expected prediction without any features
pred_xgboost_shapr <- predict(modelo6, as.matrix(test[,-10]))
pred_xgboost_shapr <- ifelse(pred_xgboost>pto_corte_jouden,1,0)

p <- mean(pred_xgboost_shapr)

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
xxx_var <- c("ind_reclamo_regularizacion" ,  "ind_retiro_regularizacion",
               "ind_refinanciamiento" ,  "ind_aviso_ficticio",
               "ind_considerar_fec" ,  "ind_limite_pago_fec",
               "ind_ffvv_vida" ,  "ind_cartera_huerfana",
               "ind_aviso_fec" ,  "mnt_prima_emitida_usd",
               "mnt_prima_cobrada_usd" ,  "mnt_prima_emitida_usd_fec",
               "mnt_prima_cobrada_usd_fec" ,  "num_cuota_pagada_ajustada_fec",
               "num_cuota_pagada_fec" ,  "mnt_prima_emitida_usd_ajustada_fec" ,
               "mnt_prima_cobrada_usd_ajustada_fec")
  
  
explanation <- explain(
  test[1:10,xxx_var],#as.matrix( test[-1:-15, xxx_var]),#x_test,
  approach = "empirical",
  explainer = explainerobj,
  prediction_zero = p
)

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)

# Plot the resulting explanations for observations 1 and 6
plot(explanation, plot_phi0 = FALSE, index_x_test = c(1, 6))



############### PARTE DE SHARP QUE SI FUNCIONABA BIEN ############
x_var <- c( "ind_reclamo_regularizacion" ,
            "ind_aviso_ficticio" ,
            "mnt_prima_cobrada_usd" ,
            "ind_ffvv_vida"
            , "ind_limite_pago_fec"
)
y_var <- "ind_bsc_fec"
#x_train <- as.matrix(Boston[-1:-6, x_var])
#y_train <- Boston[-1:-6, y_var]
#x_test <- as.matrix(Boston[1:6, x_var])
train$ind_bsc_fec <- as.numeric(as.character(train$ind_bsc_fec))
test$ind_bsc_fec <-  as.numeric(as.character(test$ind_bsc_fec))
x_train <- as.matrix(train[ x_var])
y_train <- as.matrix(train[ y_var])
x_test <- as.matrix(test[ x_var])
# Fitting a basic xgboost model_xgboost_shrap to the training data
model_xgboost_shrap <- xgboost(
  data = x_train,
  label =  y_train,
  nround = 20,
  verbose = FALSE
)
# Prepare the data for explanation
explainer <- shapr(x_train, model_xgboost_shrap)
#> The specified model_xgboost_shrap provides feature classes that are NA. The classes of data are taken as the truth.
# Specifying the phi_0, i.e. the expected prediction without any features
p <- mean(y_train)
# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
explanation <- explain(
  x_test,
  approach = "empirical",
  explainer = explainer,
  prediction_zero = p
)
# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)
plot(explanation, plot_phi0 = FALSE, index_x_test = c(1, 6))
###################################################################################
########################EXPLICABILIDAD CON LIME ###################################
###################################################################################
# REF: https://uc-r.github.io/lime
library(lime)       # ML local interpretation
library(vip)        # ML global interpretation
library(pdp)        # ML global interpretation
library(ggplot2)    # visualization pkg leveraged by above packages
library(caret)      # ML model building
library(h2o)  

train$ind_bsc_fec<- as.numeric(as.character(train$ind_bsc_fec))
# initialize h2o
h2o.init()
h2o.no_progress()


train_obs.h2o <- as.h2o(train[,-10])
local_obs.h2o <- as.h2o(train[,10])
# Create Random Forest model with ranger via caret
fit.caret <- train(
  formula, 
  data = train_obs.h2o, 
  method = 'ranger',
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
  tuneLength = 1,
  importance = 'impurity'
)

# Error in .h2o.startJar(ip = ip, port = port, name = name, nthreads = nthreads,  : 
# Your java is not supported: java version "1.7.0_45"
# Please download the latest Java SE JDK from the following URL:
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#java-requirements
