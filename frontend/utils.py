def format_response(agent, department, product):
    """
    Formats the response to display on the Streamlit app.
    """
    return (
        f"### Assigned Agent: {agent}  \n"
        f"### Department: {department}  \n"
        f"### Product: {product}"
    )
