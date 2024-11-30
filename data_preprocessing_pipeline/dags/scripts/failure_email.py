from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText
from jinja2 import Template
import logging
import os


def get_custom_logger():
    # Customer logs are stored in the below path
    log_path = os.path.join(
        os.path.dirname(__file__), "../../logs/application_logs/preprocessing_log.txt"
    )

    log_directory = os.path.dirname(log_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        print(f"Directory '{log_directory}' has been created.")

    # Create the file if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as file:
            pass  # Create an log empty file
        print(f"File '{log_path}' has been created.")

    custom_logger = logging.getLogger("preprocessing_logger")
    custom_logger.setLevel(logging.INFO)

    # Avoid default logs by setting propagate to False
    custom_logger.propagate = False

    # Set up a file handler for the custom logger
    if not custom_logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)

    return custom_logger


def send_failure_email(context):
    logger = get_custom_logger()
    sender_email = "sucessemailtrigger@gmail.com"
    receiver_email = [
        "hegde.anir@northeastern.edu",
        "nenavath.r@northeastern.edu",
        "pandey.raj@northeastern.edu",
        "khatri.say@northeastern.edu",
        "singh.arc@northeastern.edu",
        "goparaju.v@northeastern.edu",
    ]
    password = "jomnpxbfunwjgitb"

    task_instance = context["task_instance"]
    exception = context["exception"]

    subject_template = (
        "Airflow Failure: {{ task_instance.dag_id }} - {{ task_instance.task_id }}"
    )
    body_template = "The task {{ task_instance.task_id }} in DAG {{ task_instance.dag_id }} has failed. Exception: {{ exception }}"

    subject = Template(subject_template).render(task_instance=task_instance)
    body = Template(body_template).render(
        task_instance=task_instance, exception=str(exception)
    )

    email_message = MIMEText(body, "html")
    email_message["Subject"] = subject
    email_message["From"] = sender_email
    email_message["To"] = ", ".join(receiver_email)

    logger.info(
        f"Attempting to send failure email for task: {task_instance.task_id} in DAG: {task_instance.dag_id}"
    )

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        logger.info(f"Failure email sent successfully to {receiver_email}.")
    except Exception as e:
        logger.error(f"Error sending failure email: {e}")
    finally:
        server.quit()
        logger.info("SMTP server connection closed.")
