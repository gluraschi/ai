# plantilla de pre-procesado

# importar el dataset en la variable dataset
dataset = read.csv("Data.csv")

# reemplazar los valores NaN
# la función ave recibe un segundo parámetro que devuelve la media de todos los
# valores excepto los que tienen el valor NA
dataset$Age = ifelse(
  is.na(dataset$Age),
  ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), 
  dataset$Age)

# se hace lo mismo para el salario
dataset$Salary = ifelse(
  is.na(dataset$Salary),
  ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), 
  dataset$Salary)

# codificar las variables categóricas
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                         levels = c("No", "Yes"),
                         labels = c(0, 1))

# dividir los datos en conjunto de entrenamiento y conjunto de test
install.packages("caTools") # con esta línea se installa un package
library(caTools) # con esta línea se carga la librería sin el administrador de packages de la derecha
set.seed(123)
# devuelve un vector que indica qué valores son los que pertenecen al conjunto de entrenamiento
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# extrae el conjunto de entrenamiento que correspondan al vector donde los valores son true o false
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)
View(training_set)
View(testing_set)

# escalado de valores
training_set[,2:3] = scale(training_set)
testing_set[,2:3] = scale(testing_set)