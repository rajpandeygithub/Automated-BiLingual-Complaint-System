import streamlit as st
from backend import fetch_backend_response
from utils import format_response
from google.cloud import logging as gcloud_logging

# Set up Google Cloud Logging
client = gcloud_logging.Client()
logger = client.logger("complaint-portal")

# Function to reset input field
def reset_input():
    st.session_state["complaint_text"] = ""  # Clear the text field

def main():
    st.title("📢 Customer Complaint Portal")
    st.write("We value your feedback and are committed to resolving your concerns promptly. Please describe your issue below.")

    # Initialize session state
    if "complaint_text" not in st.session_state:
        st.session_state["complaint_text"] = ""

    # Multi-line input field
    st.text_area(
        "Describe your complaint:",
        value=st.session_state["complaint_text"],
        placeholder="Enter your complaint here... (max 299 words)",
        height=150,
        key="complaint_text",
    )

    # Buttons in a row
    col1, col2 = st.columns([1, 1])
    with col1:
        # Submit button (always enabled)
        submit_button = st.button("Submit")
    with col2:
        # Reset button to clear the text and state
        reset_button = st.button("Reset", on_click=reset_input)

    # Handle Submit button
    if submit_button:
        complaint_text = st.session_state["complaint_text"].strip()
        if complaint_text:
            word_count = len(complaint_text.split())
            try:
                response = fetch_backend_response(complaint_text)
                if "error" in response and "validation" in response["error"]:
                    st.error("⚠️ Your complaint must be between 6 and 299 words. Please revise your submission and try again.")
                elif "error" in response:
                    st.error(f"⚠️ {response['error']}")
                else:
                    formatted_response = format_response(
                        response.get("department", "other"),
                        response.get("product", "other"),
                    )
                    st.success("Your complaint has been registered successfully.")
                    st.markdown(formatted_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
