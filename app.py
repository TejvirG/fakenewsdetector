import streamlit as st
from model import predict

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("Fake News Detector 📰")

st.markdown("Enter a news headline below and the model will predict whether it's likely REAL or FAKE.")

headline = st.text_area("Paste a news headline", height=120)

if st.button("Check"):
    if not headline or headline.strip() == "":
        st.warning("Please paste a headline to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                label, prob = predict(headline)
                prob_pct = round(prob * 100, 1)
                if label.upper() == "REAL":
                    st.success(f"REAL")
                else:
                    st.error(f"FAKE")

                
            except Exception as e:
                st.error(f"Error running model: {e}")

st.write("---")

