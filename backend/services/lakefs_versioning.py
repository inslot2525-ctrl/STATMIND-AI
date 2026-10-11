"""
Optional lakeFS integration, switched on by LAKEFS_ENDPOINT.

lakeFS is version control for the files StatMind stores. With it on:
  * every file is written through lakeFS's S3 gateway to a branch of one
    repository (see LakeFSStorage in storage.py)
  * each dataset upload and each trained model becomes a lakeFS commit
  * datasets and models record their commit id, so "which exact data was this
    model trained on?" has a verifiable answer

Committing is best effort: if a commit fails, the file is still stored and the
problem is logged, and the id is simply missing. Only startup is strict, since
without a reachable repository nothing can be stored at all.
"""
import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

import config

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20


class LakeFSError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"lakeFS returned {status}: {message}")
        self.status = status
        self.message = message


def enabled() -> bool:
    return bool(config.LAKEFS_ENDPOINT)


def _api(method: str, path: str, body: dict = None):
    """Call the lakeFS REST API. Raises LakeFSError for HTTP errors and OSError if unreachable."""
    credentials = f"{config.LAKEFS_ACCESS_KEY_ID}:{config.LAKEFS_SECRET_ACCESS_KEY}".encode()
    request = urllib.request.Request(
        f"{config.LAKEFS_ENDPOINT.rstrip('/')}/api/v1{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": "Basic " + base64.b64encode(credentials).decode(),
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            message = json.loads(raw).get("message", raw)
        except ValueError:
            message = raw
        raise LakeFSError(error.code, str(message)[:300])


def _repository_path() -> str:
    return f"/repositories/{urllib.parse.quote(config.LAKEFS_REPOSITORY, safe='')}"


def _branch_path() -> str:
    return f"{_repository_path()}/branches/{urllib.parse.quote(config.LAKEFS_BRANCH, safe='')}"


def ensure_repository() -> None:
    """Create the repository on first use. Called at startup; raises if lakeFS is unusable."""
    try:
        _api("GET", _repository_path())
        return
    except LakeFSError as error:
        if error.status != 404:
            raise RuntimeError(f"lakeFS is not usable: {error}")
    except OSError as error:
        raise RuntimeError(f"Cannot reach lakeFS at {config.LAKEFS_ENDPOINT}: {error}")

    try:
        _api(
            "POST",
            "/repositories",
            {
                "name": config.LAKEFS_REPOSITORY,
                "storage_namespace": config.LAKEFS_STORAGE_NAMESPACE,
                "default_branch": config.LAKEFS_BRANCH,
            },
        )
        logger.info("Created lakeFS repository %s", config.LAKEFS_REPOSITORY)
    except LakeFSError as error:
        raise RuntimeError(f"Could not create lakeFS repository {config.LAKEFS_REPOSITORY}: {error}")


def head_commit() -> str | None:
    """The commit the branch currently points at."""
    if not enabled():
        return None

    try:
        return _api("GET", _branch_path())["commit_id"]
    except (LakeFSError, OSError):
        logger.exception("Could not read the lakeFS branch head")
        return None


def commit(message: str, metadata: dict = None) -> str | None:
    """
    Commit everything written to the branch since the last commit and return
    the commit id. If nothing changed (for example the same file was uploaded
    again), the current head is returned: that commit already contains the data.
    """
    if not enabled():
        return None

    body = {
        "message": message,
        "metadata": {key: str(value) for key, value in (metadata or {}).items() if value is not None},
    }

    # A commit can lose a race with another writer on the same branch; one retry settles it
    for attempt in (1, 2):
        try:
            return _api("POST", f"{_branch_path()}/commits", body)["id"]
        except LakeFSError as error:
            if error.status == 400 and "no changes" in error.message:
                return head_commit()
            if attempt == 2 or error.status not in (409, 412, 423):
                logger.error("lakeFS commit failed (%s): %s", message, error)
                return None
        except OSError:
            logger.exception("lakeFS commit failed (%s)", message)
            return None


def commit_url(commit_id: str | None) -> str | None:
    if not commit_id or not config.LAKEFS_UI_URL:
        return None
    return f"{config.LAKEFS_UI_URL.rstrip('/')}/repositories/{config.LAKEFS_REPOSITORY}/commits/{commit_id}"


def describe(dataset_commit: str | None, model_commit: str | None) -> dict | None:
    """Lineage record stored with a model and shown in the UI."""
    if not enabled():
        return None

    return {
        "repository": config.LAKEFS_REPOSITORY,
        "branch": config.LAKEFS_BRANCH,
        "dataset_commit": dataset_commit,
        "dataset_commit_url": commit_url(dataset_commit),
        "model_commit": model_commit,
        "model_commit_url": commit_url(model_commit),
    }
