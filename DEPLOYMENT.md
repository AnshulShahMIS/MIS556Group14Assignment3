# GitHub + Streamlit Community Cloud

Deploy a private source repository with public app access. Each visitor connects
their own key; do not add an operator API key to GitHub or Streamlit secrets.

The deployment folder uses `requirements-cloud.txt` as `requirements.txt`.
Choose Python 3.12 and entry point `app.py` in Streamlit's deployment form.
Keep HTTPS, XSRF protection and CORS protection enabled. The hosting service
controls the bind address and port. Local development can explicitly bind to
loopback: `streamlit run app.py --server.address 127.0.0.1`.

The 50,000-vector NPZ file contains only corpus embeddings, identifiers and model
metadata. Startup validates it against the corpus and model, then initializes
Chroma once through Streamlit's process-wide resource cache. It never contains
session records or credentials. First startup also downloads the pinned public
embedding model; no provider key is needed for initialization.

In Streamlit's Sharing settings, set app access to public so visitors do not
need GitHub accounts. A private repository does not automatically imply public
app access. Repository authorization/account consent must be completed by the
owner when prompted.

Session workspaces use random identifiers and are not served as static files.
Keys remain only in that session's memory. Run records and review queues are
temporary server-local files, and may disappear after redeploy, reboot or host
cleanup. Export work before leaving. This is a classroom demonstration, with
simulated sending and no real customer account access; use synthetic messages.

Release checks: dependency resolution, fresh vector import and same-brand
retrieval, unit/UI checks, and deployed HTTPS health plus a two-browser session
isolation check. Then use a fresh visitor key for one normal support reply and
one risky escalation. Hosting is not verified until these checks run on the
actual deployment. The earlier 200-case benchmark is not a benchmark of the
updated workflow.
