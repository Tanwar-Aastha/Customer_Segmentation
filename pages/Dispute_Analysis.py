import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import plotly.express as px

# Load Dataset
@st.cache_data
def load_data():
    file_path = "D:/Aastha/Projects/final_dispute/dataset/modified_data.csv"  # Update with correct path
    return pd.read_csv(file_path)

df = load_data()

# Page Title
st.title("Consumer Complaints Analysis")
st.write("Explore trends and insights from consumer complaint data.")

df['Date_received'] = pd.to_datetime(df['Date received'], format='%d-%m-%Y')
df['Received_year'] = pd.to_datetime(df['Date received']).dt.year
df['Date sent to company'] = pd.to_datetime(df['Date sent to company'], format='%d-%m-%Y')


# Complaint Distribution by Product
st.subheader("Complaint Distribution by Product")
fig1 = px.bar(df['Product'].value_counts().reset_index(), x='Product', y='count',
              labels={'Product': 'Product', 'count': 'Count'},
              title="Number of Complaints by Product", color='Product')
st.plotly_chart(fig1)
st.write("**Insight:** The bar chart reveals that mortgage complaints are slightly below 120,000, making them the highest among all categories. Debt collection and credit reporting follow, each with around 60,000–65,000 complaints. Credit card and bank account or service complaints are close to 40,000, indicating notable issues in banking and credit services. Other financial products like consumer loans, student loans, money transfers, payday loans, prepaid cards, and virtual currency have significantly fewer complaints. The high number of mortgage complaints suggests persistent issues in the housing finance sector, while debt collection and credit reporting remain major consumer concerns.")


# Distribution of Company Responses
st.subheader("Company Responses Distribution")

# Count occurrences of each response
response_counts = df['Company response to consumer'].value_counts().reset_index()
response_counts.columns = ['Company response', 'Count']

# Create Bar Chart
fig2 = px.bar(response_counts, 
              y='Company response', 
              x='Count', 
              color='Company response', 
              title='Company Response to Consumer')

st.plotly_chart(fig2)
st.write("**Insight:** The majority of consumer complaints (over 250K) are Closed with explanation, indicating that companies primarily respond by providing clarifications rather than offering compensation. The second most common resolution is Closed with non-monetary relief (~50K), suggesting companies sometimes address issues through policy changes or service adjustments. Closed with monetary relief (~25K) is significantly less frequent, showing that financial compensation is rare. Closed without relief (~15K) and Closed (~10K) occur less often, while Closed with relief has the lowest count (~5K), indicating that direct consumer benefits are rare suggesting a preference for explanations over actual remediation. ")


# Timely Response Effect
st.subheader("Effect of Timely Response on Disputes")
timely_effect = df.groupby('Timely response?')['Consumer disputed?'].value_counts(normalize=True).unstack().reset_index()
timely_effect.columns = ['Timely response', 'Not Disputed', 'Disputed']

timely_effect_melted = timely_effect.melt(id_vars='Timely response', 
                                          var_name='Dispute Status', 
                                          value_name='Proportion')
# Define custom colors
custom_colors = {'Not Disputed': '#FFA15A',  # Green
                 'Disputed': '#17BECF'}  # Gold
# Create Stacked Bar Chart
fig3 = px.bar(timely_effect_melted, 
              x='Timely response', 
              y='Proportion', 
              color='Dispute Status', 
              barmode='stack', 
              title='Effect of Timely Response on Disputes',
              color_discrete_map=custom_colors)
st.plotly_chart(fig3)
st.write("🔍 **Insight:** The chart highlights the critical impact of timely responses on customer disputes. When companies fail to respond on time, nearly 100% of complaints result in disputes. In contrast, when a timely response is given, disputes drop to around 40%, with the remaining 60% of complaints being resolved without a dispute. This indicates that timely responses can cut disputes by more than half, significantly improving customer satisfaction and reducing escalations. However, some disputes still occur despite prompt responses, suggesting that factors beyond response time such as resolution quality and issue complexity also play a role.")


# Complaint Submission Methods
st.subheader("Complaint Submission Methods")
fig4 = px.pie(df, names='Submitted via', title="Distribution of Submission Methods", hole=0.4)
st.plotly_chart(fig4)
st.write("🔍 **Insight:** Web submissions dominate at 65.6%, making it the most preferred method. Referrals account for 19.1%, followed by phone submissions at 7.19% and postal mail at 6.59%. Fax (1.47%) and email (0.0524%) are the least used. The overwhelming reliance on web-based submissions suggests digital channels are the primary mode of communication, while traditional methods like fax and email are nearly obsolete. ")


# 📌 Complaints Over Time
st.subheader("Complaints Over Time")
# Group by month
df['Date received'] = pd.to_datetime(df['Date received'], format='%d-%m-%Y')
time_series = df.groupby(df['Date received'].dt.to_period("M")).size().reset_index()
time_series['Date received'] = time_series['Date received'].astype(str)  # Convert period to string

# Create Line Chart
fig5 = px.line(time_series, 
               x='Date received', 
               y=0, 
               markers=True,  # Add markers on data points
               line_shape="linear", 
               title="Trends of Complaints Over Time",
               labels={'Date received': 'Month', '0': 'Number of Complaints'})

# Customize line color
fig5.update_traces(line=dict(color='crimson', width=2))
st.plotly_chart(fig5)
st.write("🔍 **Insight:** The trend of complaints over time shows a steady increase from 2011 to 2016, peaking at around 10,000 complaints per month in early 2016. Initially, complaints were below 2,000, but they surged rapidly, crossing 4,000 by mid-2012 and continuing to rise. The fluctuations between 2013 and 2015 indicate periodic increases and decreases, likely influenced by external factors such as policy changes or service quality variations. However, there is a sharp decline after mid-2016, with complaints dropping drastically to near zero. This sudden drop suggests a possible change in reporting mechanisms, data availability issues, or significant business improvements that reduced complaints.")

# Navigation to Prediction Page
st.sidebar.write("🔍 Want to predict disputes?")
if st.sidebar.button("Go to Predictor"):
    st.switch_page("pages/Dispute_Prediction.py")
