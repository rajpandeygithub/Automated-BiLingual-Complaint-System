from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import logging

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

sender_email = "sucessemailtrigger@gmail.com"
password = "jomnpxbfunwjgitb"
receiver_emails = ["hegde.anir@northeastern.edu","nenavath.r@northeastern.edu", "pandey.raj@northeastern.edu","khatri.say@northeastern.edu","singh.arc@northeastern.edu","goparaju.v@northeastern.edu"]

def send_success_email():
    logger = get_custom_logger()
    
    # Log the start of the email sending process
    logger.info("Starting to send success email to the team.")
    
    # Create the email content
    subject = 'Airflow Success: Data+Model Pipeline tasks succeeded'
    body = 'Hi team,\nThe Data+Model Pipeline tasks succeeded.'
    
    try:
        # Set up the SMTP server
        logger.info("Setting up SMTP server connection.")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        logger.info("Logged into SMTP server successfully.")
        
        # Send email to each receiver
        for receiver_email in receiver_emails:
            # Log each recipient
            logger.info(f"Sending email to {receiver_email}.")
            
            # Create a fresh message for each recipient
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            # Send the email
            server.sendmail(sender_email, receiver_email, message.as_string())
            logger.info(f"Success email sent successfully to {receiver_email}!")

    except Exception as e:
        # Log any errors encountered
        logger.error(f"Error sending success email: {e}")
    finally:
        # Close the server connection
        server.quit()
        logger.info("SMTP server connection closed.")
