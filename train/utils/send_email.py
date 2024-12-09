import smtplib
from typing import List, Dict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



# Function to send success email
def send_success_email(sender_email: str, receiver_emails: List[str], password: str, email_content: Dict[str, str]):

    # sender_email: "sucessemailtrigger@gmail.com"
    # password: "jomnpxbfunwjgitb"

    # receiver_emails = ["hegde.anir@northeastern.edu",
    #                     "nenavath.r@northeastern.edu",
    #                     "pandey.raj@northeastern.edu",
    #                     "khatri.say@northeastern.edu",
    #                     "singh.arc@northeastern.edu",
    #                     "goparaju.v@northeastern.edu"]

    # Create the email content
    # subject = '[Kubeflow Pipeline] - Started'
    # body = f'''Hi team,

    # Model training in the Kubeflow pipeline has started!

    # Please monitor the pipeline for further updates.
    # '''

    subject = email_content.get('subject')
    body = email_content.get('message')

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)

        # Send email to each receiver
        for receiver_email in receiver_emails:
            # Create a fresh message for each recipient
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            # Send the email
            server.sendmail(sender_email, receiver_email, message.as_string())

    except Exception as e:
        pass
    finally:
        server.quit()

# Function to send failure email
def send_failure_email(sender_email: str, password: str, error_message: str):

    receiver_emails = ["hegde.anir@northeastern.edu",
                        "nenavath.r@northeastern.edu",
                        "pandey.raj@northeastern.edu",
                        "khatri.say@northeastern.edu",
                        "singh.arc@northeastern.edu",
                        "goparaju.v@northeastern.edu"]

    # Create the email content
    subject = '[Kubeflow Pipeline]'
    body = f'Hi team,\nModel training has failed!.\nError Details: {error_message}'

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)

        # Send email to each receiver
        for receiver_email in receiver_emails:
            # Create a fresh message for each recipient
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            # Send the email
            server.sendmail(sender_email, receiver_email, message.as_string())

    except Exception as e:
        pass
    finally:
        server.quit()