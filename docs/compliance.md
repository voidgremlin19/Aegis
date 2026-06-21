# Project Aegis – Compliance & AI Safety Documentation

> This document describes how Project Aegis supports AI safety regulations, what
> data is collected, and how to handle personal data responsibly.

---

## Table of Contents

1. [Overview](#overview)
2. [EU AI Act Alignment](#eu-ai-act-alignment)
3. [What Data Aegis Logs](#what-data-aegis-logs)
4. [Data Retention & Deletion](#data-retention--deletion)
5. [Personal Data Handling](#personal-data-handling)
6. [Security Controls Summary](#security-controls-summary)
7. [Incident Response](#incident-response)
8. [Disclaimer](#disclaimer)

---

## Overview

Project Aegis is a **technical safety layer** designed to reduce the risk that a
deployed LLM produces misaligned, deceptive, or emotionally manipulative outputs.
It operates in-process, leaving no raw user data on external servers.

Aegis supports compliance with emerging AI safety frameworks by:

- Providing **real-time intervention** with a transparent, auditable log
- Enforcing **human oversight** via the deception tripwire (escalation to review)
- Enabling **configurable risk thresholds** appropriate to different deployment contexts
- Logging **hashed prompts** (not plaintext) by default to respect user privacy

---

## EU AI Act Alignment

The EU Artificial Intelligence Act (Regulation (EU) 2024/1689) establishes obligations
for providers and deployers of AI systems. The following table maps Aegis capabilities
to relevant articles.

| AI Act Article | Obligation | Aegis Support |
|---------------|------------|---------------|
| **Art. 9** – Risk Management | Ongoing identification and management of risks | Aegis calibration quantifies and monitors alignment-risk per prompt, logged to `audit.log` |
| **Art. 13** – Transparency | Users informed of AI nature & capabilities | Dashboard clearly labels Aegis-steered vs. raw outputs; escalation messages are explicit |
| **Art. 14** – Human Oversight | Human oversight for high-risk systems | `DeceptionTripwire` halts generation and raises an escalation flag; review queue can be attached |
| **Art. 15** – Accuracy & Robustness | Appropriate accuracy, resilience to manipulation | Threat Neutralizer and Goldilocks Tuner constrain the model's activation space to a pre-calibrated safe zone |
| **Art. 61** – Post-market Monitoring | Logging serious incidents | `audit.log` records every escalation event with timestamps, prompt hash, and reason |

> [!NOTE]
> Aegis is a technical control, not a complete compliance solution. Legal and
> organisational measures are required in addition to technical controls.

---

## What Data Aegis Logs

### Audit log (`audit.log`)

The audit log is a **rotating JSON-lines** file. Each entry contains:

| Field | Content | Privacy classification |
|-------|---------|----------------------|
| `timestamp` | UTC ISO-8601 timestamp | Non-personal |
| `event` | Event type (e.g. `ws_generate_end`) | Non-personal |
| `request_id` | UUID per request | Non-personal (pseudonymous) |
| `model` | Model name loaded | Non-personal |
| `prompt_hash` | SHA-256 of raw prompt | **Pseudonymous** – cannot be reversed without knowing the prompt |
| `prompt_preview` | First 120 chars, sanitised | **Potentially personal** if user prompt contains PII |
| `thresholds` | Config thresholds at request time | Non-personal |
| `escalated` | Boolean | Non-personal |
| `escalation_reason` | Exception message | Non-personal |
| `tokens_generated` | Count | Non-personal |

### What is NOT logged

- Full prompt text (only the first 120 sanitised characters in preview, and SHA-256 hash)
- Full model response
- User identity or IP address (IP is used for rate-limiting in-memory only; not persisted)
- Authentication credentials

### In-memory state

Activation similarities and steering vectors are held in RAM during a request and
discarded when `reset_state()` is called. They are never written to disk.

---

## Data Retention & Deletion

### Default rotation policy

The audit log rotates automatically when it reaches `audit_log_max_bytes` (default 10 MB),
keeping `audit_log_backup_count` (default 5) compressed archives.

**Total default retention: ~50 MB of JSON-lines ≈ approximately 500,000 events.**

### Configuring retention

```yaml
audit_log_path: "/var/log/aegis/audit.log"
audit_log_max_bytes: 52428800    # 50 MB per file
audit_log_backup_count: 3        # keep 3 backups → 150 MB total
```

### Manual deletion

To delete all audit data:
```bash
rm audit.log audit.log.1 audit.log.2 audit.log.3 audit.log.4 audit.log.5
```

### Responding to erasure requests (GDPR Art. 17)

Because Aegis logs **hashed** prompts, erasure of a specific user's data requires:

1. Knowing the exact prompt(s) the user submitted
2. Computing `sha256(prompt)` and searching `audit.log` for matching `prompt_hash` entries
3. Redacting those lines from the log file

A utility script (`scripts/redact_audit.py`) can be used for this purpose.

---

## Personal Data Handling

### Prompt preview (`prompt_preview`)

The first 120 characters of each prompt are stored in the audit log as a
**debugging aid**. If your deployment handles sensitive or personal data in
prompts, you should:

**Option A – Disable preview entirely:**
```yaml
# In your server.py customisation, set MAX_PREVIEW_LEN = 0
# Or set this env variable:
AEGIS_AUDIT_PREVIEW_LEN=0
```

**Option B – Apply additional masking:**
Extend `_sanitize_prompt()` in `server.py` to apply regex redaction before
the preview is written to the audit log.

### IP addresses for rate limiting

Client IP addresses are used **in-memory only** by the `slowapi` rate limiter
and are never written to disk. They are discarded when the process restarts.

### Authentication keys

The `AEGIS_API_KEY` environment variable is read at startup and compared in
constant-time (`hmac.compare_digest`). It is never logged.

---

## Security Controls Summary

| Control | Implementation | Notes |
|---------|---------------|-------|
| Authentication | `X-API-Key` header, env-var backed | No key → open mode (dev only) |
| Transport encryption | TLS via uvicorn `ssl_certfile/ssl_keyfile` | Required for production |
| Rate limiting | `slowapi` – configurable RPM per IP | Default 60 rpm |
| Prompt sanitisation | Truncate to 2000 chars, strip newlines | Applied before any logging |
| Audit logging | Rotating JSON-lines file | SHA-256 hash of full prompt |
| CORS | `CORSMiddleware`, locked to `AEGIS_ALLOWED_ORIGINS` | Default: localhost only |
| Dependency isolation | Python virtual environment | Pin versions via `pip freeze` |

---

## Incident Response

When the Deception Tripwire fires (`escalated: true` in the audit log):

1. **Automatic**: Generation halts immediately; no more tokens are produced
2. **Audit entry**: The event is logged with `escalation_reason` containing the
   exact similarity value, threshold, and token position
3. **Dashboard alert**: The frontend displays a red escalation banner
4. **Human review**: Your organisation should route escalated events to a human
   reviewer within a defined SLA

### Recommended escalation SLA (example)

| Severity | Condition | SLA |
|----------|-----------|-----|
| High | `deception_tripwire_active = true` | Review within 1 hour |
| Medium | `threat_neutralizer_active = true` on > 5 consecutive tokens | Review within 24 hours |
| Low | Any escalation in batch processing | Review within 5 business days |

---

## Disclaimer

Project Aegis is provided as-is under the MIT License. It is a **best-effort
technical control** and does not guarantee that all misaligned, harmful, or
deceptive outputs will be detected or prevented. Users are responsible for:

- Conducting appropriate risk assessments for their specific deployment context
- Implementing organisational safeguards alongside technical controls
- Complying with all applicable laws and regulations in their jurisdiction
- Not relying solely on Aegis for high-stakes safety-critical applications

The cosine-similarity thresholds used by Aegis are probabilistic: a sufficiently
novel or adversarially crafted prompt may evade detection. Regular re-calibration
and red-teaming are strongly recommended.
