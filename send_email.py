import smtplib
from email.mime.text import MIMEText

# Create the message
msg = MIMEText("hello world")
msg['Subject'] = "Hello from Python"
msg['From'] = "sabin.maharjan364@gmail.com"
msg['To'] = "sabin.maharjan456@gmail.com"

# Send the email
with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
    smtp.starttls()
    smtp.login('sabin.maharjan364@gmail.com', 'pajp xhqc jojn ukgk')
    smtp.send_message(msg)
    print("Email sent successfully!")