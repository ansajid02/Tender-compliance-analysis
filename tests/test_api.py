"""HTTP API end to end: auth, project isolation, uploads + jobs, check, exports, citation viewer, ask (SSE)."""

import json
import os

import pytest
from fakes import FakeLLM
from fastapi.testclient import TestClient

from agentcore import llm
from tender import db, jobs, samples


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return samples.write_all(tmp_path_factory.mktemp("samples"))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setitem(llm.BACKENDS, "gemini", FakeLLM())
    from tender.api import app

    with TestClient(app) as c:
        yield c


def keys():
    _, alice = db.create_user("alice")
    _, bob = db.create_user("bob")
    _, admin = db.create_user("root", is_admin=True)
    return {"alice": {"X-API-Key": alice}, "bob": {"X-API-Key": bob}, "admin": {"Authorization": f"Bearer {admin}"}}


def wait_ok(job_id: str) -> dict:
    job = jobs.wait(job_id, timeout=60)
    assert job.status == "done", job.error
    return job.to_dict()


def make_checked_project(client, files, h) -> str:
    pid = client.post("/projects", json={"name": "Cloud RFP"}, headers=h).json()["project_id"]
    for role, key in (("tender", "tender"), ("bid", "bid")):
        r = client.post(f"/projects/{pid}/documents", data={"role": role}, headers=h,
                        files=[("files", (files[key].name, files[key].read_bytes()))])
        assert r.status_code == 202, r.text
        wait_ok(r.json()["accepted"][0]["job_id"])
    r = client.post(f"/projects/{pid}/check", json={"budget_usd": 1.0}, headers=h)
    assert r.status_code == 202
    job = wait_ok(r.json()["job_id"])
    assert job["result"]["judged"] > 10 and job["stage"] == "done"
    return pid


def test_health_needs_no_key_and_reports_dependencies(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["database"] and r["vector_store"]
    assert client.get("/").status_code == 200 and "Tender Compliance Checker" in client.get("/").text


def test_auth_required_and_invalid_keys_rejected(client):
    assert client.get("/projects").status_code == 401
    assert client.get("/projects", headers={"X-API-Key": "tk_wrong"}).status_code == 401
    assert client.get("/projects", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_keys_are_stored_hashed_and_can_be_revoked(client):
    _, key = db.create_user("carol")
    with db.session() as s:
        stored = [k.key_hash for k in s.query(db.ApiKey).all()]
    assert key not in stored and db.hash_key(key) in stored
    assert client.get("/me", headers={"X-API-Key": key}).json()["name"] == "carol"
    assert db.revoke_key(key[:10]) == 1
    assert client.get("/me", headers={"X-API-Key": key}).status_code == 401


def test_projects_are_isolated_between_users(client, files):
    h = keys()
    pid = client.post("/projects", json={"name": "Alice's tender"}, headers=h["alice"]).json()["project_id"]
    assert [p["project_id"] for p in client.get("/projects", headers=h["alice"]).json()] == [pid]
    assert client.get("/projects", headers=h["bob"]).json() == []
    for method, path in (("get", f"/projects/{pid}"), ("get", f"/projects/{pid}/matrix"),
                         ("delete", f"/projects/{pid}"), ("post", f"/projects/{pid}/check")):
        assert getattr(client, method)(path, headers=h["bob"]).status_code == 404  # not 403: existence hidden
    assert client.get(f"/projects/{pid}", headers=h["admin"]).status_code == 200
    db.grant(pid, "bob")
    assert client.get(f"/projects/{pid}", headers=h["bob"]).status_code == 200


def test_upload_rejects_bad_files_before_any_work(client):
    h = keys()["alice"]
    pid = client.post("/projects", json={"name": "x"}, headers=h).json()["project_id"]
    r = client.post(f"/projects/{pid}/documents", data={"role": "bid"}, headers=h,
                    files=[("files", ("evil.pdf", b"<script>not a pdf</script>"))])
    assert r.status_code == 400 and "not a PDF" in r.text
    r = client.post(f"/projects/{pid}/documents", data={"role": "judge"}, headers=h,
                    files=[("files", ("a.pdf", b"%PDF-1.7"))])
    assert r.status_code == 422  # role must be tender | bid
    assert client.post(f"/projects/{pid}/check", headers=h).status_code == 400  # no documents yet


def test_full_flow_check_exports_viewer_audit(client, files):
    h = keys()["alice"]
    pid = make_checked_project(client, files, h)
    detail = client.get(f"/projects/{pid}", headers=h).json()
    assert {d["role"] for d in detail["documents"]} == {"tender", "bid"}
    assert detail["latest_check"]["judged"] > 10 and detail["jobs"][0]["kind"] == "check"

    m = client.get(f"/projects/{pid}/matrix", headers=h).json()
    row = next(r for r in m["matrix"] if r["bid_quote"])
    assert row["tender_doc_id"] and row["bid_doc_id"] and m["documents"][row["bid_doc_id"]] == "bid_acme.docx"
    csv = client.get(f"/projects/{pid}/matrix?format=csv", headers=h)
    assert csv.headers["content-type"].startswith("text/csv") and "attachment" in csv.headers["content-disposition"]
    xlsx = client.get(f"/projects/{pid}/matrix?format=xlsx", headers=h)
    assert xlsx.content[:2] == b"PK"
    assert "Compliance check" in client.get(f"/projects/{pid}/summary", headers=h).text

    # citation viewer: PDF page as PNG with the quote highlighted; DOCX page text with the quote's span
    img = client.get(f"/projects/{pid}/documents/{row['tender_doc_id']}/pages/{row['tender_page']}/image",
                     params={"q": row["tender_quote"]}, headers=h)
    assert img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"
    assert int(img.headers["x-highlights"]) >= 1
    txt = client.get(f"/projects/{pid}/documents/{row['bid_doc_id']}/pages/{row['bid_page']}/text",
                     params={"q": row["bid_quote"]}, headers=h).json()
    a, b = txt["highlight"]
    assert txt["text"][a:b] == row["bid_quote"]
    assert client.get(f"/projects/{pid}/documents/{row['bid_doc_id']}/pages/99/text", headers=h).status_code == 404
    assert client.get(f"/projects/{pid}/documents/d-nope/file", headers=h).status_code == 404

    # ask: Server-Sent Events, answer only after citation verification
    with client.stream("POST", f"/projects/{pid}/ask", json={"question": "How many days are backups kept?"},
                       headers=h) as r:
        body = "".join(r.iter_text())
    events = [blk.split("\n")[0].removeprefix("event: ") for blk in body.strip().split("\n\n")]
    assert events[0] == "status" and "answer" in events and events[-1] == "done"
    answer = json.loads(next(b for b in body.split("\n\n") if b.startswith("event: answer")).split("data: ", 1)[1])
    assert answer["citations"] and "120 days" in answer["answer"]

    # delete: everything gone, but the audit trail survives in the DB
    assert client.delete(f"/projects/{pid}", headers=h).json() == {"deleted": pid}
    assert client.get(f"/projects/{pid}", headers=h).status_code == 404
    actions = [e["action"] for e in db.audit_log(pid)]
    assert actions[0] == "delete_project" and "check" in actions and "upload" in actions


def test_check_budget_cap_stops_a_runaway_job(client, files, monkeypatch):
    """Each fake call 'costs' $0.01; a $0.02 budget lets extraction run, then stops the judges."""
    h = keys()["alice"]
    pid = client.post("/projects", json={"name": "budget"}, headers=h).json()["project_id"]
    for role in ("tender", "bid"):
        r = client.post(f"/projects/{pid}/documents", data={"role": role}, headers=h,
                        files=[("files", (files[role].name, files[role].read_bytes()))])
        wait_ok(r.json()["accepted"][0]["job_id"])
    monkeypatch.setattr(llm, "cost_usd", lambda *a: 0.01)
    job_id = client.post(f"/projects/{pid}/check", json={"budget_usd": 0.02}, headers=h).json()["job_id"]
    job = jobs.wait(job_id)
    assert job.status == "failed" and "budget" in job.error


def test_job_status_is_private(client, files):
    h = keys()
    pid = client.post("/projects", json={"name": "p"}, headers=h["alice"]).json()["project_id"]
    r = client.post(f"/projects/{pid}/documents", data={"role": "tender"}, headers=h["alice"],
                    files=[("files", (files["tender"].name, files["tender"].read_bytes()))])
    job_id = r.json()["accepted"][0]["job_id"]
    wait_ok(job_id)
    assert client.get(f"/jobs/{job_id}", headers=h["bob"]).status_code == 404
    assert client.get(f"/jobs/{job_id}", headers=h["alice"]).json()["result"]["pages"] == 6


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test on Postgres")
def test_postgres_backend(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    import uuid

    name = f"pg-{uuid.uuid4().hex[:6]}"
    user, key = db.create_user(name)
    assert db.authenticate(key).name == name and db.ping()
    job = db.new_job("check", "p-pgtest", user, {"sample": 3})
    db.update_job(job.id, status="done", result={"ok": True})
    assert db.get_job(job.id).result == {"ok": True}
    db.audit(name, "test", "p-pgtest", x=1)
    assert db.audit_log("p-pgtest")[0]["detail"] == {"x": 1}


def test_ui_references_only_existing_elements_and_never_uses_innerhtml():
    import re
    from pathlib import Path

    web = Path(__file__).parents[1] / "tender" / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    js = "\n".join(line for line in js.splitlines() if not line.lstrip().startswith("//"))  # code, not comments
    ids_used = set(re.findall(r'\$\("([A-Za-z]+)"\)', js))
    ids_defined = set(re.findall(r'id="([A-Za-z]+)"', html))
    assert ids_used and ids_used <= ids_defined, ids_used - ids_defined
    assert "innerHTML" not in js and "outerHTML" not in js and "insertAdjacentHTML" not in js  # untrusted text
