import streamlit as st
from streamlit_chat import message
from rag import rag
from postgres import insert_dialog, insert_feedback
import uuid

def main():

    # Initialize session state if not already done
    if "uuid" not in st.session_state:
        st.session_state["uuid"] = ""
    if "i" not in st.session_state:
        st.session_state["i"] = 0
    if "count" not in st.session_state:
        st.session_state["count"] = 0
    if "responses" not in st.session_state:
        st.session_state["responses"] = []
    if "thumbsup_disabled" not in st.session_state:
        st.session_state["thumbsup_disabled"] = True
    if "thumbsdown_disabled" not in st.session_state:
        st.session_state["thumbsdown_disabled"] = True
    if "feedback_given" not in st.session_state:
        st.session_state["feedback_given"] = False

    # Function to handle the dialo input
    def handle_input():
        # Get user input
        user_input = st.session_state["user_input"]
        
        # Check session state
        if user_input:
            st.session_state["responses"].append(("user", user_input))
            st.session_state["thumbsup_disabled"] = True
            st.session_state["thumbsdown_disabled"] = True
            st.session_state["feedback_given"] = False
            st.session_state["user_input"] = ""
            
            with input_con:
                # Show spinner
                st.spinner("Thinking...")

                # Get response
                response = rag(user_input, model='gpt-4o-mini')
            
            # Create a unique UUID
            st.session_state["uuid"] = str(uuid.uuid4())

            # Store dialog to database
            insert_dialog(st.session_state["uuid"], user_input, response)
            
            st.session_state["responses"].append(("system", response["answer"]))
            st.session_state["thumbsup_disabled"] = False
            st.session_state["thumbsdown_disabled"] = False

    # Function to handle positive feedback
    def give_positive_feedback():
        
        # Increase count
        st.session_state["count"] += 1
        # Store positive feedback in database
        insert_feedback(st.session_state["uuid"], 1)
        
        if not st.session_state["thumbsup_disabled"]:
            st.session_state["feedback_given"] = True
            st.session_state["thumbsup_disabled"] = True
            st.session_state["thumbsdown_disabled"] = True

    # Function to handle negative feedback
    def give_negative_feedback():
        
        # Decrease count
        st.session_state["count"] -= 1
        # Store negative feedback in database
        insert_feedback(st.session_state["uuid"], -1)

        if not st.session_state["thumbsdown_disabled"]:
            st.session_state["feedback_given"] = True
            st.session_state["thumbsdown_disabled"] = True
            st.session_state["thumbsup_disabled"] = True

    # Chat interface

    # Set browser tab title
    st.set_page_config(page_title="eCommerce site assistant", menu_items=None, page_icon="random")

    # Set page title
    st.title('eCommerce site assistant')

    # Display select message
    st.sidebar.success("Select an option above.")

    # Container label
    st.write("Dialog window")

    # Messages container
    messages_con = st.container(border=True, height=300)

    # Count container
    count_con = st.container(border=True, height=70)
    with count_con:
        count_text = st.text(f"Count: {st.session_state.count}")

    # Input container
    input_con = st.container(border=True)
    with input_con:
        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
        with col1:
            st.button(":thumbsup:", disabled=st.session_state["thumbsup_disabled"], on_click=give_positive_feedback)
        with col2:
            st.button(":thumbsdown:", disabled=st.session_state["thumbsdown_disabled"], on_click=give_negative_feedback)
        with col3:
            st.text_input("Ask a question:", key="user_input", on_change=handle_input)

    # Display chat messages
    with messages_con:
        for sender, msg in st.session_state["responses"]:
            key_name = "key" + str(st.session_state["i"])
            st.session_state["i"] += 1
            message(msg, key=key_name, is_user=(sender == "user"))

if __name__ == "__main__":
    main()
    
