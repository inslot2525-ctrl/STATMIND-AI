"""
Admin commands.

    python manage.py create-user EMAIL [--admin]
    python manage.py claim-legacy-data EMAIL
    python manage.py migrate

`claim-legacy-data` gives every dataset, model and experiment created before
accounts existed (i.e. with no owner) to the given user.
"""
import argparse
import getpass
import sys

from sqlalchemy import update

from database import Dataset, Experiment, ModelRegistry, SessionLocal, User, init_db
from services.auth import create_user, normalize_email


def cmd_create_user(args):
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    try:
        user = create_user(args.email, password, is_admin=args.admin)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Created {'admin ' if user.is_admin else ''}user {user.email} ({user.id})")


def cmd_claim_legacy(args):
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=normalize_email(args.email)).first()
        if user is None:
            sys.exit(f"No user with email {args.email}. Create it first with create-user.")
        counts = {}
        for model in (Dataset, Experiment, ModelRegistry):
            result = db.execute(update(model).where(model.owner_id.is_(None)).values(owner_id=user.id))
            counts[model.__tablename__] = result.rowcount
        db.commit()
    finally:
        db.close()
    print("Assigned to", args.email + ":", ", ".join(f"{n} {t}" for t, n in counts.items()))


def main():
    parser = argparse.ArgumentParser(description="StatMind admin commands")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-user", help="Create a user account")
    p.add_argument("email")
    p.add_argument("--admin", action="store_true")
    p.set_defaults(func=cmd_create_user)

    p = sub.add_parser("claim-legacy-data", help="Give unowned (pre-accounts) data to a user")
    p.add_argument("email")
    p.set_defaults(func=cmd_claim_legacy)

    p = sub.add_parser("migrate", help="Upgrade the database schema")
    p.set_defaults(func=lambda _args: print("Database is up to date."))

    args = parser.parse_args()
    init_db()
    args.func(args)


if __name__ == "__main__":
    main()
