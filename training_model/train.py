from sklearn import metrics
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.feature import(OneHotEncoder, StringIndexer)
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.getOrCreate()
spark
data = spark.read.csv("./adult.csv", inferSchema= True, header= True)
data.printSchema()
data_new = data.select('age','workclass','education','occupation','gender',
                          'capital-gain','capital-loss',
                          'hours-per-week','income')


data_new = data_new.dropna(how = "any")

df = data_new.toPandas()

num_var = [i for (i,dt) in data_new.dtypes if (dt == "int" and i != "income")]
cat_var = [i for (i,dt) in data_new.dtypes if dt == "string"]

str_idx = [StringIndexer(inputCol = c, outputCol = c+"_StringIndexer", handleInvalid="skip")for c in cat_var]
oh_enc = [OneHotEncoder(inputCols = [f"{c}_StringIndexer"], outputCols = [f"{c}_OneHotEncoder"]) for c in cat_var]
agg_ip  = [c for c in num_var]
agg_ip = agg_ip + [c+"_OneHotEncoder" for c in cat_var]
vect_assembler = VectorAssembler(inputCols =  agg_ip, outputCol = "Vect_features")
steps = []
steps += str_idx
steps += oh_enc
steps += [vect_assembler]
ppline = Pipeline().setStages(steps)
model = ppline.fit(data_new)
model.save("pipeline_model")
pred = model.transform(data_new)
dta = pred.select(F.col("Vect_features").alias("features"), F.col("income").alias("label"),)

train, test = dta.randomSplit([0.7,0.3], seed = 1)
print("train: ", train.count(), "\ntest: ", test.count())
test.groupBy("label").count().show()

model = LogisticRegression().fit(train)
model_test = model.transform(test)

# claculating the accuracy, precision, recall and f1_score
actual = model_test.select(['label']).collect()
predicted = model_test.select(['prediction']).collect()
acc = metrics.accuracy_score(actual, predicted)
prec = metrics.precision_score(actual, predicted)
recall = metrics.recall_score(actual, predicted)
f1 = metrics.f1_score(actual, predicted)
print("accuracy: ", acc, "\nprecision: ", prec, "\nrecall: ", recall, "\nf1_score: ", f1)


# save the model
# pickle.dump(model, open("model.pkl", "wb"))
# model = pickle.load(open("model.pkl", "rb"))

model.save("logistic_regression_model")

