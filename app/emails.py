"""Sending sign-in links. The console sender is for development and tests."""

import smtplib
import sys
from abc import ABC, abstractmethod
from email.message import EmailMessage
from typing import TextIO

from app.config import Settings


class EmailSender(ABC):
    @abstractmethod
    def send(self, to: str, subject: str, body: str) -> None:
        """Send one plain-text email."""


class ConsoleEmailSender(EmailSender):
    """Prints emails instead of sending them, and keeps them in outbox for tests.

    This prints to the console, not to the logs, so links never reach log files.
    """

    def __init__(self, stream: TextIO | None = sys.stdout) -> None:
        self.stream = stream
        self.outbox: list[dict[str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.outbox.append({"to": to, "subject": subject, "body": body})
        if self.stream is not None:
            print(f"--- Email to {to}: {subject}\n{body}\n---", file=self.stream, flush=True)


class SMTPEmailSender(EmailSender):
    def __init__(self, settings: Settings) -> None:
        if not settings.smtp_host or not settings.smtp_from:
            raise ValueError("SMTP_HOST and SMTP_FROM are required when EMAIL_SENDER=smtp")
        self.settings = settings

    def send(self, to: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.settings.smtp_from
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=30) as smtp:
            if self.settings.smtp_starttls:
                smtp.starttls()
            if self.settings.smtp_username:
                smtp.login(self.settings.smtp_username, self.settings.smtp_password)
            smtp.send_message(message)


def email_sender_for(settings: Settings) -> EmailSender:
    if settings.email_sender == "console":
        return ConsoleEmailSender()
    if settings.email_sender == "smtp":
        return SMTPEmailSender(settings)
    raise ValueError(f"EMAIL_SENDER must be console or smtp, got {settings.email_sender!r}")
