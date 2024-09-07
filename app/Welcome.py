import streamlit as st
from es import init_es
from postgres import init_postgres

def main():

     # Elastic Search initialization not done yet
    if "init_es_done" not in st.session_state or not st.session_state.init_es_done:
        init_es()
        st.session_state.init_es_done = True

    # Postgres initialization not done yet
    if "init_postgres_done" not in st.session_state or not st.session_state.init_postgres_done:
        init_postgres()
        st.session_state.init_postgres_done = True   

    # Set browser tab title
    st.set_page_config(page_title="Welcome to the eCommerce site assistant", menu_items=None, page_icon="random")

    # Set page title
    st.title('Welcome to the eCommerce site assistant')

    # Display select message
    st.sidebar.success("Select an option above.")

    # Display info
    st.markdown(
        """This is the main page of the eCommerce site assistant application. You can select an option 
        from the sidebar on the left side. The available options are:
        """
    )
    st.markdown("- Assistant: On this page you can ask your question and get answers from the assistant. You can also leave a positive or negative feedback.")
    st.markdown("- RAG evaluation: On this page you can perform the RAG evaluation.")
    st.markdown("- Retrieval evaluation: On this page you can perform the retrieval evaluation.")

if __name__ == "__main__":
    main()