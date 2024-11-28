import logging
import streamlit as st
from google.cloud import logging as gcloud_logging
from backend import fetch_backend_response
from utils import format_response

# Initialize Google Cloud Logging client
gcloud_client = gcloud_logging.Client()
gcloud_client.setup_logging()  # Sends logs to Google Cloud Logging

# Function to log word violations
def log_word_violation(word_count, complaint_text):
    if word_count < 6 or word_count > 299:
        logging.warning(f"Word count violation: {word_count} words. Complaint preview: {complaint_text[:100]}...")

# Function to reset input field
def reset_input():
    st.session_state["complaint_text"] = ""  # Clear the text field
    st.session_state["submit_enabled"] = False  # Reset the Submit button state

def main():
    st.title("📢 Customer Complaint Portal")
    st.write("We value your feedback and are committed to resolving your concerns promptly. Please describe your issue below.")

    # Initialize session state
    if "complaint_text" not in st.session_state:
        st.session_state["complaint_text"] = ""
    if "submit_enabled" not in st.session_state:
        st.session_state["submit_enabled"] = True  # Submit always enabled (no min word restriction)

    # Multi-line input field
    st.text_area(
        "Describe your complaint:",
        value=st.session_state["complaint_text"],
        placeholder="Enter your complaint here... (max 299 words)",
        height=150,
        key="complaint_text"
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
        word_count = len(st.session_state["complaint_text"].split())
        log_word_violation(word_count, st.session_state["complaint_text"])  # Log violations

        if st.session_state["complaint_text"].strip():
            try:
                response = fetch_backend_response(st.session_state["complaint_text"])
                if "error" in response and "validation" in response["error"]:
                    st.error("⚠️ Maximum word limit is 299. Please shorten your complaint.")
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
