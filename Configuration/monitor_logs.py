import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import glob
import os

def send_email(subject, body, to_emails, from_email, from_password):
    """
    Send an email using the provided credentials.
    """
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject

        # Attach the body
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server
        server = smtplib.SMTP('instaltic-com.correoseguro.dinaserver.com', 587)  # Use STARTTLS on port 587
        server.starttls()  # Upgrade to a secure connection
        server.login(from_email, from_password)

        # Send the email
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def monitor_logs():
    """
    Monitor log files for the keyword "Not connected" and send email notifications.
    """
    # Target log file pattern
    log_pattern = "/home/dolph_user/data/?/Dolph/log/Dolph.log"

    # Email credentials
    to_emails = ["", ""]
    from_email = ""
    from_password = ""

    # Subject and email body
    subject = "Alert: 'Not connected' found in logs"

    # Check each log file
    for log_file in glob.glob(log_pattern):
        try:
            with open(log_file, 'r') as file:
                lines = file.readlines()

            # Look for the keyword "Not connected"
            error_lines = [line for line in lines if "Not connected" in line]

            if error_lines:
                body = f"The following errors were found in {log_file}:\n\n"
                body += "\n".join(error_lines)

                # Send the email
                send_email(subject, body, to_emails, from_email, from_password)
        except Exception as e:
            print(f"Failed to process {log_file}: {e}")

if __name__ == "__main__":
    monitor_logs()
