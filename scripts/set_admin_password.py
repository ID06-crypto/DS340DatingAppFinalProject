"""Make someone an administrator and set their password.

    python scripts/set_admin_password.py someone@bu.edu

It asks for the password twice, without showing it. To pipe it in instead, send it on
standard input. It uses the same settings as the server (APP_MAIN_DB and so on), creates
the account if it doesn't exist, and makes it an approved admin. The password is stored
only as a scrypt hash. Admins can change it later on the admin Password page.

This is the only way to make an admin: joining on the site never does, because nothing
checks that people own the email they give. On Render, run it in the service's Shell tab.
"""

import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.security import hash_password, normalize_email, password_problem  # noqa: E402
from app.config import Settings  # noqa: E402
from live_store import LiveStore  # noqa: E402
from manager import ProfileManager  # noqa: E402


def read_password() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("New password: ")
    if getpass.getpass("Again: ") != first:
        sys.exit("The passwords don't match.")
    return first


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    settings = Settings.from_env()
    email = normalize_email(sys.argv[1], settings.allowed_email_domains)
    if email is None:
        sys.exit("That isn't a valid email address.")
    listed = {normalize_email(e) for e in settings.admin_emails}
    if settings.production and email not in listed:
        sys.exit(f"{email} isn't in ADMIN_EMAILS. Add it there first, so the admins are written down.")
    password = read_password()
    problem = password_problem(password)
    if problem:
        sys.exit(problem)
    ProfileManager(settings.main_db, settings.persona_db)  # Creates the tables if needed.
    live = LiveStore(settings.main_db)
    account, created = live.get_or_create_account(email, admin=True)
    if account.role != "admin" or account.status != "approved":
        live.make_admin(account.account_id, "set_admin_password")
    live.set_password_hash(account.account_id, hash_password(password))
    live.audit("script", "password_change", account.account_id, {"made_admin": account.role != "admin", "created": created})
    print(f"{email} is an administrator, with a password set.{' (New account.)' if created else ''}")
    if account.user_id is not None:
        print(f"Note: {email} also has a participant profile, so other participants will see and rate it.")


if __name__ == "__main__":
    main()
