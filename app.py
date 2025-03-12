from flask import Flask, request, render_template
import pandas as pd

from sklearn.preprocessing import StandardScaler
from customer_segmentation.pipeline.predict_pipeline import CustomDataClass, PredictPipelineClass


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomDataClass(
            Date_Received=request.form.get('Date_Received'),
            Product_Name=request.form.get('Product_Name'),
            Issue_Detail=request.form.get('Issue_Detail'),
            Submitted_Via=request.form.get('Submitted_Via'),
            Date_sent_to_company=request.form.get('Date_sent_to_company'),
            Company_Response=request.form.get('Company_Response'),
            Timely_Response=request.form.get('Timely_Response')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipelineClass()
        result = predict_pipeline.predict(pred_df)

        return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
