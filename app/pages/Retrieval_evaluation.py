import streamlit as st
from eval import generate_ground_truth_data, evaluate_text_search_retrieval, evaluate_vector_search_retrieval

def main():

    # Set session state
    if "text_eval" not in st.session_state:
        st.session_state["text_eval"] = ""

    if "vector_search_q_eval" not in st.session_state:
        st.session_state["vector_search_q_eval"] = ""

    if "vector_search_a_eval" not in st.session_state:
        st.session_state["vector_search_a_eval"] = ""

    if "vector_search_qa_eval" not in st.session_state:
        st.session_state["vector_search_qa_eval"] = ""

    # Set browser tab title
    st.set_page_config(page_title="eCommerce site assistant retrieval evaluation ", menu_items=None, page_icon="random")

    # Set page title
    st.title('eCommerce site assistant retrieval evaluation')

    # Display select message
    st.sidebar.success("Select an option above.")

    # Ground truth data
    st.markdown("### Ground truth data generation")
    gt_btn_clicked = st.button("Click to generate ground truth data")

    # If button is clicked
    if gt_btn_clicked:
        with st.spinner("Generating..."):
            generate_ground_truth_data()

    # Show a red message to inform user that files are already generated
    st.markdown(":red[File has already been generated.]")

    # Text search evaluation
    st.markdown("### Text search retrieval evaluation")

    # Click button to calculate evaluation
    text_btn_clicked = st.button("Click to evaluate text search retrieval")
    
    # If button is clicked
    if text_btn_clicked:
        with st.spinner("Calculating ..."):
            hit_rate, mrr = evaluate_text_search_retrieval()
            st.session_state["text_eval"] = f"Hit Rate: {hit_rate}, MRR: {mrr}"
            
            # Display Metrics
            st.write(st.session_state["text_eval"])
 
    # Vector search evaluation
    st.markdown("### Vector search retrieval evaluation")

    # Click button to calculate evaluations
    vector_btn_clicked = st.button("Click to evaluate vector search retrieval")

    # If btn_clicked
    if vector_btn_clicked:
        with st.spinner("Calculating ..."):
            # Question
            hit_rate, mrr = evaluate_vector_search_retrieval("question_vector")
            st.session_state["vector_search_q_eval"] = f"Question vector: Hit Rate: {hit_rate}, MRR: {mrr}"
            # Answer
            hit_rate, mrr = evaluate_vector_search_retrieval("answer_vector")
            st.session_state["vector_search_a_eval"] = f"Answer vector: Hit Rate: {hit_rate}, MRR: {mrr}"
            # Question - Answer
            hit_rate, mrr = evaluate_vector_search_retrieval("question_answer_vector")
            st.session_state["vector_search_qa_eval"] = f"Question - Answer vector: Hit Rate: {hit_rate}, MRR: {mrr}"
            
            # Display Metrics
            st.write(st.session_state["vector_search_q_eval"])
            st.write(st.session_state["vector_search_a_eval"])
            st.write(st.session_state["vector_search_qa_eval"])
 
if __name__ == "__main__":
    main()