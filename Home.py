import streamlit as st

# Set page title
st.set_page_config(page_title="Consumer Dispute Prediction", layout="wide")
# st.sidebar.success("Select a demo above.")


# Title and Introduction
st.title("🔍 Consumer Dispute Prediction App")
# st.write("""
# This web application helps predict whether a **consumer complaint** will result in a dispute.  
# It utilizes **machine learning** to analyze various complaint factors such as **response time, issue type, and company actions**.
# """)
st.write("""
### What is a Customer Dispute?
A **customer dispute** occurs when a consumer disagrees with a business regarding a product, service, or transaction. These disputes may arise due to billing errors, unsatisfactory services, product defects, or unfulfilled promises. If unresolved, disputes can escalate into complaints, negative reviews, or even legal action.
         """)

# Customer-dispute Picture
st.image("customer_dispute.jpg")

st.write("""
### Why is it Important for Businesses?
Understanding and predicting customer disputes is **crucial** for businesses because:
- It helps improve **customer satisfaction** by addressing issues proactively.
- It reduces **financial losses** from refunds, chargebacks, and legal claims.
- It enhances **brand reputation** by minimizing negative feedback and complaints.
- It allows companies to refine their **customer support and dispute resolution** strategies.

### What Does This Web App Do?
This web application helps predict whether a **consumer complaint** will result in a dispute.  
It utilizes **machine learning** to analyze various complaint factors such as **response time, issue type, and company actions**.  
By leveraging historical data, businesses can identify potential disputes early and take proactive measures to resolve them efficiently.
""")


# Sections
st.subheader("🔹 Features of This App:")
st.markdown("""
- **Prediction Page:** Enter complaint details to predict if the consumer will dispute or not.
- **Analysis Page:** Explore various insights and visualizations from the complaint dataset.
""")


# Add some space before the footer
# st.markdown("---")

