from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText
from jinja2 import Template
import logging
import os

# Defining a custom logger
def get_custom_logger():
    # Customer logs are stored in the below path
    log_path = os.path.join(os.path.dirname(__file__), "../../logs/application_logs/preprocessing_log.txt")
    custom_logger = logging.getLogger("preprocessing_logger")
    custom_logger.setLevel(logging.INFO)
    
    # Avoid default logs by setting propagate to False
    custom_logger.propagate = False

    # Set up a file handler for the custom logger
    if not custom_logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
        file_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)
    
    return custom_logger

def send_failure_email(task_instance, exception):
    logger = get_custom_logger()
    
    sender_email = "sucessemailtrigger@gmail.com"
    receiver_email = ["hegde.anir@northeastern.edu","nenavath.r@northeastern.edu", "pandey.raj@northeastern.edu","khatri.say@northeastern.edu","singh.arc@northeastern.edu","goparaju.v@northeastern.edu"]
    password = 'jomnpxbfunwjgitb'

    # Subject and body for the failure email
    subject_template = 'Airflow Failure: {{ task_instance.dag_id }} - {{ task_instance.task_id }}'
    body_template = 'The task {{ task_instance.task_id }} in DAG {{ task_instance.dag_id }} has failed. Exception: {{ exception }}'

    # Render templates using Jinja2 Template
    subject = Template(subject_template).render(task_instance=task_instance)
    body = Template(body_template).render(task_instance=task_instance, exception=str(exception))

    # Create the email headers and content
    email_message = MIMEText(body, 'html')
    email_message['Subject'] = subject
    email_message['From'] = sender_email
    email_message['To'] = receiver_email

    # Log the email sending attempt
    logger.info(f"Attempting to send failure email for task: {task_instance.task_id} in DAG: {task_instance.dag_id}")

    try:
        # Set up the SMTP server
        logger.info("Setting up SMTP server connection.")
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        logger.info("Logged into SMTP server successfully.")
        
        # Send the email
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        logger.info(f"Failure email sent successfully to {receiver_email}.")

    except Exception as e:
        # Log any errors encountered
        logger.error(f"Error sending failure email: {e}")

    finally:
        # Close the server connection
        server.quit()
        logger.info("SMTP server connection closed.")
