def format_response(department, product):
    """
    Formats the response to display on the Streamlit app.
    """
    return (
        f"It has been assigned to **Agent_Placeholder** in the **{department}** department. "
        f"Our team will review the issue related to **{product}** shortly. "
        f"Thank you for your patience."
    )
