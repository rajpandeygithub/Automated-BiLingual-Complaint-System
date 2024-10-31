# scripts/Success_email.py
def send_success_email():
    sender_email = Variable.get('EMAIL_USER')
    password = Variable.get('EMAIL_PASSWORD')
    receiver_emails = [
        "hegde.anir@northeastern.edu",
        "nenavath.r@northeastern.edu",
        "pandey.raj@northeastern.edu",
        "khatri.say@northeastern.edu",
        "singh.arc@northeastern.edu",
        "goparaju.v@northeastern.edu"
    ]

    # Define subject and body
    subject = 'Airflow Success: Data+Model Pipeline tasks succeeded'
    body = '''Hi team,
    The Data+Model Pipeline tasks succeeded.'''

    # Create the email headers and content
    email_message = MIMEMultipart()
    email_message['Subject'] = subject
    email_message['From'] = sender_email
    email_message['To'] = ", ".join(receiver_emails)
    email_message.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        
        # Send email to each receiver
        for receiver_email in receiver_emails:
            email_message.replace_header('To', receiver_email)
            server.sendmail(sender_email, receiver_email, email_message.as_string())
            print(f"Success email sent successfully to {receiver_email}!")

    except Exception as e:
        print(f"Error sending success email: {e}")
    finally:
        server.quit()