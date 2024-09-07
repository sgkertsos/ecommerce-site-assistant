import streamlit as st
from eval import calculate_similarities
import seaborn as sns
import matplotlib.pyplot as plt
from eval import generate_offline_rag_evaluation_data
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Grafana URL
GRAFANA_URL = os.getenv("GRAFANA_URL")

def main():
    # Set browser tab title
    st.set_page_config(page_title="eCommerce site assistant RAG evaluation ", menu_items=None, page_icon="random")

    # Set page title
    st.title('eCommerce site assistant RAG evaluation')

    # Display select message
    st.sidebar.success("Select an option above.")

    # Offline RAG evaluation
    st.markdown("### Offline RAG evaluation")
    
    # Evaluation files generation
    st.markdown("##### Evaluation files generation ")
    # Click button to generate RAG evaluation files    
    offline_btn_clicked = st.button("Click to generate files for offline RAG evaluation")

    # If button is clicked
    if offline_btn_clicked:
        with st.spinner("Generating file for gpt-3.5-turbo..."):
            generate_offline_rag_evaluation_data(model="gpt-3.5-turbo")
        with st.spinner("Generating file for gpt-4o-mini..."):
            generate_offline_rag_evaluation_data(model="gpt-4o-mini")

    # Show a red message to inform user that files are already generated
    st.markdown(":red[Files have already been generated.]")

    # Cosine similarity calculation
    st.markdown("##### Cosine Similarity")

    # Click button to calculate cosine similarity 
    cosine_btn_clicked = st.button("Click to calculate cosine similarity")
    if cosine_btn_clicked:
        # Create 2 columns
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown("##### gpt-3.5-turbo")
            with st.spinner('Calculating...'):
                # Calculate similarities
                similarities_35, df_35 = calculate_similarities(model='gpt-3.5-turbo')

            # Show values
            st.write(f"Count: {similarities_35['count']}")
            st.write(f"Mean: {similarities_35['mean']}")
            st.write(f"Std: {similarities_35['std']}")
            st.write(f"Min: {similarities_35['min']}")
            st.write(f"25%: {similarities_35['25%']}")
            st.write(f"50%: {similarities_35['50%']}")
            st.write(f"75%: {similarities_35['75%']}")
            st.write(f"Max: {similarities_35['max']}")

            # Display graph
            sns.displot(df_35)
            st.pyplot(plt.gcf())

        with col2:
            st.markdown("##### gpt-4o-mini")
            with st.spinner("Calculating..."):
                # Calculate similarities
                similarities_4o, df_4o = calculate_similarities(model='gpt-4o-mini')

            # Show values
            st.write(f"Count: {similarities_4o['count']}")
            st.write(f"Mean: {similarities_4o['mean']}")
            st.write(f"Std: {similarities_4o['std']}")
            st.write(f"Min: {similarities_4o['min']}")
            st.write(f"25%: {similarities_4o['25%']}")
            st.write(f"50%: {similarities_4o['50%']}")
            st.write(f"75%: {similarities_4o['75%']}")
            st.write(f"Max: {similarities_4o['max']}")

                # Display graph
            sns.displot(df_4o)
            st.pyplot(plt.gcf())

    # Online RAG evaluation
    st.markdown("### Online RAG evaluation")
    st.write("For online evaluation we have used User Feedback and Relevance.")
    st.write("Follow the instructions to access Grafana and watch the Feedback and Relevance Spread panels updating while a user interacts with the assistant.")

if __name__ == "__main__":
    main()
