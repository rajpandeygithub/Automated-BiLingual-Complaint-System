import streamlit as st
from backend import fetch_backend_response
from utils import format_response

def main():
    st.title("📢 Customer Complaint Portal")
    st.write("We value your feedback and are committed to resolving your concerns promptly. Please describe your issue below.")

    # Input text field
    complaint_text = st.text_area(
        "Describe your complaint:", 
        placeholder="Enter your complaint here...",
        height=150,
    )

    # submit button
    if st.button("Submit"):
        if complaint_text.strip():
            try:
                response = fetch_backend_response(complaint_text)
                if "error" in response:
                    st.error(response["error"])
                else:
                    formatted_response = format_response(
                        response.get("agent", "Agent1"),
                        response.get("product_department", "Department XYZ"),
                        response.get("product", "Product PQR"),
                    )
                    st.success("Your complaint has been registered successfully.")
                    st.markdown(formatted_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a complaint before submitting.")

if __name__ == "__main__":
    main()
