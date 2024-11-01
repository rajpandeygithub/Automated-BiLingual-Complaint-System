from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText
from jinja2 import Template

def send_failure_email(task_instance, exception):
    sender_email = "sucessemailtrigger@gmail.com"
    receiver_email = "hegde.anir@northeastern.edu"
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

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        print("Failure email sent successfully!")
    except Exception as e:
        print(f"Error sending failure email: {e}")
    finally:
        server.quit()
