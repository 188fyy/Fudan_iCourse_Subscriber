"""Read a fresh UIS verification code from a configured 163 inbox."""

from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
import imaplib
import os
import re
import ssl
import time


class EmailVerificationError(RuntimeError):
    """An email challenge needs attention before another login attempt."""


_CODE = re.compile(
    r"(?:验证码|verification\s+code)\s*(?:为|是|is)?\s*[:：]?\s*"
    r"([0-9]{4,8})(?![0-9])",
    re.IGNORECASE,
)
_SENDER = "id@fudan.edu.cn"
_SUBJECT = "用户名、短信验证码认证（Web）通知"


def _extract_code(raw: bytes) -> str | None:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if parseaddr(str(message.get("From", "")))[1].casefold() != _SENDER:
        return None
    subject = str(message.get("Subject", ""))
    if subject.strip() != _SUBJECT:
        return None
    body = message.get_body(preferencelist=("plain", "html"))
    content = body.get_content() if body else ""
    if body and body.get_content_type() == "text/html":
        content = unescape(re.sub(r"<[^>]*>", " ", content))
    codes = set(_CODE.findall(subject + "\n" + content))
    if len(codes) > 1:
        raise EmailVerificationError("UIS mail contains multiple possible verification codes")
    return next(iter(codes), None)


@contextmanager
def open_verification_inbox():
    username = os.getenv("ICOURSE_163_EMAIL", "").strip()
    password = os.getenv("ICOURSE_163_IMAP_AUTH_CODE", "").strip()
    if not username.lower().endswith("@163.com") or not password:
        raise EmailVerificationError(
            "Configure ICOURSE_163_EMAIL and ICOURSE_163_IMAP_AUTH_CODE "
            "before using UIS email verification"
        )
    try:
        with imaplib.IMAP4_SSL(
            "imap.163.com", 993, ssl_context=ssl.create_default_context(), timeout=15
        ) as inbox:
            inbox.login(username, password)
            if "ID" in inbox.capabilities:
                inbox.xatom("ID", '("name" "iCourse Subscriber" "version" "1.0")')
            status, _ = inbox.select("INBOX", readonly=True)
            if status != "OK":
                raise EmailVerificationError("Cannot open the 163 inbox")
            yield inbox
    except (imaplib.IMAP4.error, OSError):
        raise EmailVerificationError(
            "163 IMAP connection failed; check that IMAP is enabled and the authorization code is valid"
        ) from None


def inbox_checkpoint(inbox) -> tuple[int, int]:
    status, _ = inbox.noop()
    if status != "OK":
        raise EmailVerificationError("Cannot refresh the 163 inbox")
    # STATUS reads the current UID boundary even when the folder was already selected.
    status, records = inbox.status("INBOX", "(UIDVALIDITY UIDNEXT)")
    if status == "OK" and records and isinstance(records[0], bytes):
        validity = re.search(rb"\bUIDVALIDITY (\d+)", records[0])
        next_uid = re.search(rb"\bUIDNEXT (\d+)", records[0])
        if validity and next_uid:
            return int(validity[1]), int(next_uid[1])
    raise EmailVerificationError("Cannot establish the verification mail UID checkpoint")


def wait_for_verification_code(inbox, checkpoint: tuple[int, int]) -> str:
    validity, first_uid = checkpoint
    deadline = time.monotonic() + 90
    checked = set()
    while time.monotonic() < deadline:
        current_validity, _ = inbox_checkpoint(inbox)
        if current_validity != validity:
            raise EmailVerificationError("163 inbox changed while waiting for the verification mail")
        status, records = inbox.uid("SEARCH", None, "UID", f"{first_uid}:*", "FROM", f'"{_SENDER}"')
        if status != "OK":
            raise EmailVerificationError("Cannot search the 163 inbox")
        codes = []
        for uid in (records[0] or b"").split():
            if int(uid) < first_uid or uid in checked:
                continue
            status, parts = inbox.uid("FETCH", uid, "(BODY.PEEK[])")
            if status != "OK":
                raise EmailVerificationError("Cannot read the UIS verification mail")
            raw = next((part[1] for part in parts if isinstance(part, tuple)), None)
            if raw is not None:
                code = _extract_code(raw)
                if code:
                    codes.append(code)
                checked.add(uid)
        if len(codes) > 1:
            raise EmailVerificationError("Multiple new UIS verification emails; retry with one login running")
        if codes:
            return codes[0]
        time.sleep(3)
    raise EmailVerificationError("Timed out waiting for a new UIS verification email in the 163 inbox")
