import streamlit as st
from backend import fetch_backend_response
from utils import format_response

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
        st.session_state["submit_enabled"] = False

    # Function to update the Submit button state dynamically
    def update_submit_state():
        word_count = len(st.session_state["complaint_text"].split())
        st.session_state["submit_enabled"] = word_count > 5

    # Multi-line input field
    st.text_area(
        "Describe your complaint:",
        value=st.session_state["complaint_text"],
        placeholder="Enter your complaint here... (min 6 and max 299 words)",
        height=150,
        key="complaint_text",
        on_change=update_submit_state,  # Dynamically check word count
    )

    # Buttons in a row
    col1, col2 = st.columns([1, 1])
    with col1:
        # Submit button (enabled based on word count)
        submit_button = st.button("Submit", disabled=not st.session_state["submit_enabled"])
    with col2:
        # Reset button to clear the text and state
        reset_button = st.button("Reset", on_click=reset_input)

    # Handle Submit button
    if submit_button:
        if st.session_state["complaint_text"].strip():
            try:
                response = fetch_backend_response(st.session_state["complaint_text"])
                if "error" in response and "validation" in response["error"]:
                    st.error("⚠️ Maximum word limit is 299. Please shorten your complaint.")
                elif "error" in response:
                    st.error(f"⚠️ {response['error']}")
                else:
                    formatted_response = format_response(
                        #response.get("processed_text", "other"),
                        response.get("department", "other"),
                        response.get("product", "other"),
                    )
                    st.success("Your complaint has been registered successfully.")
                    st.markdown(formatted_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    
