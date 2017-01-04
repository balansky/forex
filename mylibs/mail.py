from email.mime.text import MIMEText
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage


class MailServer:

    def __init__(self, host, port, account,password,to_address):
        self.server_addr = host
        self.server_port = port
        self.acc_user = account
        self.acc_pswd = password
        self.to_address = to_address

    def send(self,subject,lines,file=None):
        # fromaddr = from_address
        toaddr = ", ".join(self.to_address)
        msg = MIMEMultipart()
        # msg.preamble = "This is a test email"
        msg['From'] = self.acc_user
        msg['To'] = toaddr
        msg['Subject'] = subject
        # msg['Cc'] = ", ".join(cc_address)
        msg.attach(MIMEText(lines,'plain'))
        if file is not None:
            attachment = self.attachment_reader(file)
            msg.attach(attachment)
        server = smtplib.SMTP_SSL(self.server_addr, self.server_port)
        server.ehlo()
        # server.starttls()
        server.login(self.acc_user, self.acc_pswd)
        text = msg.as_string()
        server.sendmail(self.acc_user,self.to_address,text)
        server.quit()

    def attachment_reader(self,file):
        ctype, encoding = mimetypes.guess_type(file)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        if maintype == "text":
            fp = open(file)
            # Note: we should handle calculating the charset
            attachment = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "image":
            fp = open(file, "rb")
            attachment = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "audio":
            fp = open(file, "rb")
            attachment = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(file, "rb")
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(fp.read())
            fp.close()
            encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename=file)
        return attachment