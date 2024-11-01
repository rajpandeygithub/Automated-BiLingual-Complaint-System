from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "sucessemailtrigger@gmail.com"
password = "jomnpxbfunwjgitb"
receiver_emails = ["hegde.anir@northeastern.edu","nenavath.r@northeastern.edu", "pandey.raj@northeastern.edu","khatri.say@northeastern.edu","singh.arc@northeastern.edu","goparaju.v@northeastern.edu"]

def send_success_email():
    # Create the email content
    subject = 'Airflow Success: Data+Model Pipeline tasks succeeded'
    body = 'Hi team,\nThe Data+Model Pipeline tasks succeeded.'
    
    # Debug print statements
    print(f"Attempting to send email from: {sender_email} to: {', '.join(receiver_emails)}")
    print(f"SMTP server: smtp.gmail.com, port: 587")

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
            print(f"Success email sent successfully to {receiver_email}!")

    except Exception as e:
        print(f"Error sending success email: {e}")
    finally:
        server.quit()