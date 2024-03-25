## Customer Dispute Segmentation

### Overview

Customer disputes arise when consumers report issues with financial products or services. Addressing these disputes is crucial for maintaining trust and regulatory compliance. This project aims to analyze customer complaints and classify them to help businesses understand common issues and improve customer satisfaction.

### Project Objective

The goal of this project is to classify customer disputes based on complaint data using a Decision Tree Classifier. By analyzing patterns in consumer complaints, the project helps in identifying key concerns and improving response strategies.

### Dataset

The dataset contains the following columns:

- `Date sent to company:` The date when the complaint was sent to the company.
- `Product:` The type of financial product or service involved in the complaint.

- `Issue:` A description of the specific problem reported by the consumer.

- `Submitted via:` The medium through which the complaint was submitted (e.g., web, phone, email, etc.).

- `Date received:` The date when the complaint was received by the regulatory body.

- `Company response to consumer:` The type of response given by the company to address the complaint.

- `Timely response?:` Indicates whether the company responded within the required timeframe (Yes/No).

- `Consumer disputed?:` Specifies whether the consumer disputed the resolution provided by the company (Yes/No).

### Approach

Since the dataset contains a dependent variable (Consumer disputed?), classification techniques were applied instead of clustering. The Decision Tree Classifier was used for segmentation.

### Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit

### Installation

To set up the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/Tanwar-Aastha/Customer_Segmentation.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```

### Usage

Run the App:
```
streamlit run Home.py
```

### View App
Access the deployed app here: [Customer Dispute Predictor App Link](https://customersegmentation-f82qy4qucfyuv2jfzkqeep.streamlit.app/)

### Results

The classification model provides insights into dispute resolution trends, helping companies enhance customer service strategies.

