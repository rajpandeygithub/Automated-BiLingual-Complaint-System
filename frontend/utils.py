import random
def format_response(department, product):
    """
    Formats the response to display on the Streamlit app.
    """
    agent = f"AgentID:{random.randint(100000, 999999)}"
    return (
        f"It has been assigned to **{agent}** in the **{department}** department. "
        f"Our team will review the issue related to **{product}** shortly. "
        f"Thank you for your patience."
    )

