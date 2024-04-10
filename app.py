import numpy as np
from flask import Flask, request, render_template
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F



app = Flask(__name__)

spark = SparkSession.builder.appName("web-app").getOrCreate()
main_model = LogisticRegressionModel.load("models/logistic_regression_model")
pipeline_model = PipelineModel.load("models/pipeline_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data_values = list(request.form.values())
    column_names = ['age', 'workclass', 'education', 'occupation', 'gender',
                'capital-gain', 'capital-loss', 'hours-per-week', 'income']


    data_dict = dict(zip(column_names, data_values))

    df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    data_new = spark.createDataFrame(df)
    integer_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']
    for col_name in integer_columns:
        data_new = data_new.withColumn(col_name, col(col_name).cast("int"))
    pred = pipeline_model.transform(data_new)
    print(pred)
    dta = pred.select(F.col("Vect_features").alias("features"), F.col("income").alias("label"),)
    print(dta)
    predictions = main_model.transform(dta)
    output = predictions.select("prediction").collect()[0][0]
     
    if(output == 0):
        output = "Below poverty Line"
    else:
        output = "Above poverty Line"
    


    return render_template('index.html', prediction_text='According to your inserted data your are {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, port=8000)
