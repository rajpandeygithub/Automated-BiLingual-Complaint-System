import streamlit as st
from backend import fetch_backend_response
from utils import format_response
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get complaint constraints
COMPLAINT_MIN_LENGTH = int(os.getenv("COMPLAINT_MIN_LENGTH", 6))
COMPLAINT_MAX_LENGTH = int(os.getenv("COMPLAINT_MAX_LENGTH", 299))

# Function to reset input field
def reset_input():
    st.session_state["complaint_text"] = ""  # Clear the text field
    st.session_state["complaint_submitted"] = False  # Reset submission flag

def show_complaint_portal():
    st.title("📢 Customer Complaint Portal")
    st.write("We value your feedback and are committed to resolving your concerns promptly. Please describe your issue below.")

    # Add a "Back to Homepage" link
    homepage_url = "https://storage.googleapis.com/frontend_homepage/homepage.html"
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <a href="{homepage_url}" target="_self" style="color: #0056e0; font-size: 1.2rem; text-decoration: none; border: 2px solid #0056e0; padding: 10px 15px; border-radius: 5px; display: inline-block; font-weight: bold;">
                ⬅️ Back to Homepage
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "complaint_text" not in st.session_state:
        st.session_state["complaint_text"] = ""
    if "complaint_submitted" not in st.session_state:
        st.session_state["complaint_submitted"] = False

    # Create a placeholder for the text area
    text_area_placeholder = st.empty()

    # If complaint is not yet submitted, show the text area
    if not st.session_state["complaint_submitted"]:
        with text_area_placeholder.container():
            # Multi-line input field
            st.text_area(
                "Describe your complaint:",
                value=st.session_state["complaint_text"],
                placeholder=f"Enter your complaint here... (min {COMPLAINT_MIN_LENGTH} and max {COMPLAINT_MAX_LENGTH} words)",
                height=150,
                key="complaint_text"
            )

    # Buttons in a row
    col1, col2 = st.columns([1, 1])
    with col1:
        # Submit button (enabled based on word count)
        submit_button = st.button("Submit")
    with col2:
        # Reset button to clear the text and state
        reset_button = st.button("Reset", on_click=reset_input)

    # Handle Submit button
    if submit_button:
        if st.session_state["complaint_text"].strip():
            try:
                response = fetch_backend_response(st.session_state["complaint_text"])
                if "error" in response and "validation" in response["error"]:
                    st.error("⚠️ Your complaint must be between 6 and 299 words. Please revise your complaint and try again")
                elif "error" in response:
                    st.error(f"⚠️ {response['error']}")
                else:
                    formatted_response = format_response(
                        response.get("department", "other"),
                        response.get("product", "other"),
                    )
                    st.success("Your complaint has been registered successfully.")
                    st.markdown(formatted_response)

                    # Mark complaint as submitted to hide the text area
                    st.session_state["complaint_submitted"] = True
                    # Clear the placeholder for the text area after submission
                    text_area_placeholder.empty()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("⚠️ Complaint text cannot be empty.")

if __name__ == "__main__":
    show_complaint_portal()
