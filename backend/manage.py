"""
Admin commands.

    python manage.py create-user EMAIL [--admin]
    python manage.py claim-legacy-data EMAIL
    python manage.py migrate
    python manage.py import-local-files

`claim-legacy-data` gives every dataset, model and experiment created before
accounts existed (i.e. with no owner) to the given user.

`import-local-files` is for switching an existing installation from local file
storage to S3 or lakeFS: it copies every dataset, model and artifact that is
still only on local disk into the configured storage. With lakeFS the import is
committed, and datasets that had no commit recorded get that one.
"""
import argparse
import getpass
import sys

from sqlalchemy import update

from database import Artifact, Dataset, Experiment, ModelRegistry, SessionLocal, User, init_db
from services import lakefs_versioning
from services.auth import create_user, normalize_email
from services.storage import LocalStorage, get_storage


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


def cmd_import_local_files(_args):
    if lakefs_versioning.enabled():
        lakefs_versioning.ensure_repository()

    storage = get_storage()
    if storage.name == "local":
        sys.exit("Storage is local, so there is nothing to import. Set LAKEFS_ENDPOINT or STORAGE_BACKEND=s3 first.")

    local = LocalStorage()
    copied, missing = 0, []

    db = SessionLocal()
    try:
        keys = {
            row.storage_key
            for model in (Dataset, ModelRegistry, Artifact)
            for row in db.query(model).filter(model.storage_key.isnot(None)).all()
        }

        for key in sorted(keys):
            if storage.exists(key):
                continue
            if not local.exists(key):
                missing.append(key)
                continue
            storage.put_file(key, local.local_path(key))
            copied += 1

        print(f"Copied {copied} file(s) into {storage.name} storage; {len(keys) - copied - len(missing)} were already there.")
        if missing:
            print(f"{len(missing)} file(s) are recorded in the database but not on local disk, for example {missing[0]}")

        if lakefs_versioning.enabled():
            commit_id = lakefs_versioning.commit(
                "Import files from local storage", {"statmind.kind": "import", "statmind.files": copied}
            )
            if commit_id:
                updated = db.execute(
                    update(Dataset).where(Dataset.data_commit.is_(None)).values(data_commit=commit_id)
                ).rowcount
                db.commit()
                print(f"lakeFS commit {commit_id[:12]}; recorded on {updated} dataset(s) that had none.")
    finally:
        db.close()


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

    p = sub.add_parser("import-local-files", help="Copy locally stored files into S3 / lakeFS storage")
    p.set_defaults(func=cmd_import_local_files)

    p = sub.add_parser("migrate", help="Upgrade the database schema")
    p.set_defaults(func=lambda _args: print("Database is up to date."))

    args = parser.parse_args()
    init_db()
    args.func(args)


if __name__ == "__main__":
    main()
