# Deployment Troubleshooting

This document catalogs real issues encountered during PolicyPulse Cloud Run deployment and their verified fixes. Maintained for future reference.

---

## Issue 1: `gcloud builds submit --dockerfile` flag does not exist

**Symptom:** 
```
ERROR: (gcloud.builds.submit) unrecognized arguments: --dockerfile=backend.Dockerfile
```

**Root cause:**  
The `--dockerfile` flag shown in some third-party blog posts is not a real flag in the current gcloud CLI (580.0.0). The official `gcloud builds submit --tag` command only accepts a file literally named `Dockerfile` with no way to override the filename via flag.

**Fix:**  
Create per-service `cloudbuild.yaml` config files (`cloudbuild.backend.yaml`, `cloudbuild.frontend.yaml`) with explicit `docker build -f <dockerfile>` steps, then use:
```bash
gcloud builds submit . --config=cloudbuild.backend.yaml --project=<project>
```

---

## Issue 2: Frontend cannot reach internal-ingress backend despite correct IAM

**Symptom:**  
Frontend Cloud Run service (with `--vpc-egress=all-traffic`) fails to reach the internal-ingress backend with:
```
httpx.ConnectError: [Errno 110] Connection timed out
```
Backend logs show zero incoming requests. IAM `roles/run.invoker` binding is correctly configured.

**Root cause:**  
Cloud Run Direct VPC Egress requires **Private Google Access (PGA)** enabled on the subnet. Without PGA, VPC-routed traffic cannot reach Google-managed services (like internal Cloud Run endpoints), even with correct IAM and network topology.

**Fix:**  
Enable Private Google Access on the subnet:
```bash
gcloud compute networks subnets update default \
  --region=us-central1 \
  --enable-private-ip-google-access \
  --project=<project>
```
After applying this, frontend → backend requests succeed immediately with no code changes.

---

## Issue 3: `gemini-2.5-flash` returns 404 for new API keys

**Symptom:**  
Gemini API returns:
```
404 models/gemini-2.5-flash is no longer available to new users
```
This is **not** a rate limit, quota issue, or code bug.

**Root cause:**  
Google deprecated `gemini-2.5-flash` for API keys created under newer/fresh Google Cloud projects or AI Studio accounts. Existing keys from older projects may still work, but new keys do not have access.

**Fix:**  
Migrate to `gemini-3.6-flash` (Gemini 3.x models), which is available to all users. Requires SDK migration (see Issue 4).

---

## Issue 4: Gemini 3.x function calling requires `thought_signature` preservation

**Symptom:**  
Agent requests with `gemini-3.6-flash` fail with:
```
400 BadRequest: Function call is missing a thought_signature in functionCall parts.
```

**Root cause (actual):**  
Gemini 3.x models require `thought_signature` metadata to be preserved across multi-turn tool-calling conversations. The **legacy `google-generativeai` SDK does not handle this automatically**. Attempting manual serialization with `MessageToDict()` or reconstructing response objects failed repeatedly because the root issue was using the wrong library, not a serialization bug.

**Fix:**  
Migrate from `google-generativeai` (deprecated) to the unified `google-genai` SDK (v2.18.1+):
- Install: `pip install google-genai` (remove `google-generativeai`)
- Update imports: `from google import genai` instead of `import google.generativeai as genai`
- Use `genai.Client(api_key=...)` instead of `genai.configure()`
- Call `client.models.generate_content(model=..., contents=..., config=...)`
- **Critical:** Append the raw SDK `Content` object from `response.candidates[0].content` to conversation history unmodified—do not manually reconstruct it. The SDK handles `thought_signature` preservation automatically **only if the raw object is used**.

**Files changed:**
- `app/services/llm_factory.py` — `GeminiLLMClient` refactored to use `genai.Client()`
- `app/services/embedding_factory.py` — `EmbeddingService` refactored to use `genai.Client().models.embed_content()`
- `app/agents/schemas.py` — `ChatMessage` and `LLMCompletionResult` store `gemini_raw_content: Any` (the SDK `Content` object) instead of serialized dicts; Pydantic configured with `arbitrary_types_allowed=True`
- `app/agents/orchestrator.py` — Updated to pass `gemini_raw_content` through message history
- `requirements.txt` — Replaced `google-generativeai` with `google-genai`

---

## Issue 5: Secret Manager values include trailing newline from Windows pipes

**Symptom:**
1. **Demo password:** User enters correct password in Streamlit UI, but authentication fails silently—no error, just re-prompts for password.
2. **Gemini API key:** Backend startup succeeds, but first Gemini API call fails with:
   ```
   grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
   status = StatusCode.UNAVAILABLE
   details = "Illegal metadata"
   ```

**Root cause:**  
Secret values created via Windows CMD/PowerShell pipe commands like:
```powershell
echo "my-password" | gcloud secrets create ...
```
silently append a trailing newline (`\n`) or carriage return + newline (`\r\n`) to the secret value. This breaks:
- Exact-match string comparisons (demo password check fails)
- gRPC header validation (API key with trailing newline is malformed)

**Fix:**  
On Windows, use `Set-Content -NoNewline` to write the secret to a temporary file, then pass the file path:
```powershell
"my-clean-secret" | Set-Content -Path temp-secret.txt -NoNewline
gcloud secrets create my-secret --data-file=temp-secret.txt --project=<project>
Remove-Item temp-secret.txt
```
**Never use pipes (`echo ... |`) on Windows for secret creation.**

For existing secrets with trailing newlines, create a new version using the file method above. Verify correct length in Secret Manager UI (e.g., Gemini API keys should be exactly 53 characters for `AIza...` format, not 54 or 55).

---

## Verification Commands

**Check Private Google Access:**
```bash
gcloud compute networks subnets describe default --region=us-central1 --project=<project> --format="value(privateIpGoogleAccess)"
```
Should return `True`.

**Check IAM invoker bindings:**
```bash
gcloud run services get-iam-policy <backend-service> --region=us-central1 --project=<project>
```
Should list both frontend SA and scheduler SA as members with `roles/run.invoker`.

**Check secret version length:**
```bash
gcloud secrets versions access latest --secret=<secret-name> --project=<project> | wc -c
```
Compare to expected length (no trailing whitespace).

**Test internal-ingress backend (should fail from public internet):**
```bash
curl https://<backend-url>/metadata
```
Should return `403 Forbidden` or `404 Not Found` (confirms internal-only ingress).

---

*Last updated: 2026-08-18 (Phase 6 Cloud Run deployment complete)*
