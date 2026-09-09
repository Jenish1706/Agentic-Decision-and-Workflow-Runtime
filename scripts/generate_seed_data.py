"""Generates the synthetic 'Nimbus' engineering knowledge base used by this
project: markdown documents (data/documents/), a structured service/team
registry (data/records.json), and a hand-authored evaluation benchmark
(data/benchmark.json).

This is a one-off content generator, not part of the runtime. It exists so
the corpus and its ground truth are reproducible and so reviewers can see
exactly how the "domain" (a fictional internal platform team) was authored.
Run: python scripts/generate_seed_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"

# ---------------------------------------------------------------------------
# Documents: (doc_id, title, category, tags, body)
# ---------------------------------------------------------------------------

DOCS = [
    ("arch-aurora", "Aurora Architecture Overview", "architecture", "aurora,ingestion,rate-limiting", """
Aurora is Nimbus's event ingestion service, owned by team Nimbus-Core. Clients
send events over HTTP or gRPC; Aurora buffers incoming events in an in-memory
ring buffer before handing them off to Cascade for downstream processing.

To protect the ring buffer from a single noisy client, Aurora enforces a
default rate limit of 5,000 events per second per tenant. Requests that
exceed this limit receive an HTTP 429 response with a Retry-After header.
The limit is configurable per tenant but requires a change request.

Aurora tracks ring buffer occupancy as a percentage; sustained occupancy
above 80% is considered a warning sign of an upstream slowdown or a tenant
exceeding its allotted throughput, and is the leading indicator used in the
Aurora throttling runbook.
"""),
    ("arch-compass", "Compass Architecture Overview", "architecture", "compass,gateway,routing", """
Compass is Nimbus's API gateway, owned by team Nimbus-Core. Every external
request to a Nimbus service is routed through Compass, which terminates TLS,
applies a global default rate limit of 10,000 requests per second, and
forwards the request to the appropriate backend service based on path-based
routing rules.

Before forwarding a request, Compass validates the caller's access token by
calling Anchor's token introspection endpoint. If the token is missing,
expired, or invalid, Compass rejects the request with a 401 before it ever
reaches the backend service. This means backend services generally do not
need to re-implement authentication themselves.
"""),
    ("arch-anchor", "Anchor Architecture Overview", "architecture", "anchor,auth,tokens", """
Anchor is Nimbus's authentication and identity service, owned by team
Nimbus-Platform. It issues short-lived JSON Web Token (JWT) access tokens
with a 15 minute time-to-live, and longer-lived refresh tokens valid for 30
days. Access tokens are validated statelessly by services; refresh tokens
are checked against Anchor's session store, an in-memory cache with periodic
snapshots to disk for recovery.

Because access tokens expire after only 15 minutes, any client that expects
to make requests over a longer session must use its refresh token to obtain
a new access token before the old one expires.
"""),
    ("arch-beacon", "Beacon Architecture Overview", "architecture", "beacon,monitoring,alerting", """
Beacon is Nimbus's monitoring and alerting service, owned by team
Nimbus-Platform. It scrapes metrics from every registered Nimbus service
every 30 seconds, evaluates a set of alert rules against the scraped data,
and pages the on-call engineer through Lighthouse when a rule's threshold is
breached.

Beacon does not deliver pages itself; delivery (SMS, push notification, or
email) is handled entirely by Lighthouse, keeping the alerting-rules engine
and the notification-delivery mechanism as separate concerns.
"""),
    ("arch-lighthouse", "Lighthouse Architecture Overview", "architecture", "lighthouse,notifications,delivery", """
Lighthouse is Nimbus's notification delivery service, owned by team
Nimbus-Platform. It delivers messages across three channels: email, SMS, and
push notification. When a delivery attempt fails, Lighthouse retries up to 3
times using exponential backoff before giving up and moving the message to a
dead-letter queue for manual inspection.

Lighthouse is used both for on-call paging (triggered by Beacon) and for
product-facing notifications triggered by other services through its public
API.
"""),
    ("arch-cascade", "Cascade Architecture Overview", "architecture", "cascade,etl,pipeline", """
Cascade is Nimbus's batch data pipeline, owned by team Nimbus-Data. It reads
events written by Aurora, runs a set of transformation jobs every 15 minutes,
and writes the transformed output to Ferry for long-term storage.

Cascade's job state and scheduling metadata are stored in a Postgres
database with a hot standby replica for failover. Because Cascade jobs run
on a 15 minute cycle, a short failover event is generally absorbed by the
next scheduled run without any manual intervention beyond what the failover
runbook describes.
"""),
    ("arch-ferry", "Ferry Architecture Overview", "architecture", "ferry,storage,objects", """
Ferry is Nimbus's object storage service, owned by team Nimbus-Data. It
stores blobs with configurable per-bucket retention policies and supports
object versioning. Ferry uses tiered storage: recently written or frequently
accessed objects live in a "hot" tier, while older, rarely accessed objects
are migrated to a cheaper "cold" tier by a background job.

Buckets have a storage quota; once a bucket approaches its quota, the
storage cleanup runbook describes how to reclaim space, either by migrating
more aggressively to cold storage or by deleting objects that are no longer
needed.
"""),
    ("runbook-aurora-throttling", "Runbook: Aurora Throttling / 429s", "runbook", "aurora,throttling,incident", """
Use this runbook when Aurora starts returning 429 responses to a tenant, or
when Beacon alerts that Aurora's ring buffer occupancy is above 80%.

Steps:
1. Check the tenant's current request rate against its configured limit
   (default 5,000 events/sec/tenant).
2. Check ring buffer occupancy. If it is climbing even though the offending
   tenant is within its limit, the issue may be a downstream slowdown in
   Cascade rather than the tenant itself.
3. If the tenant has a legitimate, temporary need for a higher limit, raise
   the per-tenant limit via the Aurora config and notify the tenant.
4. If occupancy remains high after addressing the tenant's rate, escalate to
   the Nimbus-Core on-call engineer following the on-call escalation runbook.
"""),
    ("runbook-deploy-rollback", "Runbook: Deployment Rollback", "runbook", "deploy,rollback,operations", """
This is the standard rollback procedure for any Nimbus service.

Steps:
1. Identify the last known-good deployment version from the deploy tool's
   history.
2. Run `deploy rollback <service> <version>` using the deploy tool.
3. A rollback requires at least 1 approver, same as a forward deploy.
4. Post a notification in the #deploys channel describing what was rolled
   back and why.
5. Confirm service health in Beacon before closing out the rollback.

Rollback does not require a full incident review unless the original bad
deploy caused customer-visible impact, in which case a postmortem is
expected.
"""),
    ("runbook-db-failover", "Runbook: Database Failover", "runbook", "cascade,anchor,database,failover", """
This runbook covers failing over a Postgres primary to its replica for
either Cascade or Anchor, the two services backed by Postgres.

Expected recovery targets: Recovery Time Objective (RTO) of 5 minutes and
Recovery Point Objective (RPO) of 30 seconds. If a failover is taking
noticeably longer than 5 minutes, escalate immediately rather than waiting.

Steps:
1. Confirm the primary is actually unreachable (not just slow) using Beacon.
2. Promote the replica using the database tool's `promote` command.
3. Update the service's connection configuration to point at the new
   primary, if not handled automatically by service discovery.
4. Verify writes are succeeding again before declaring the failover
   complete.
5. The old primary, once recovered, rejoins the cluster as a replica; it
   must not be reintroduced as primary without a manual review.
"""),
    ("runbook-oncall-escalation", "Runbook: On-Call Escalation", "runbook", "oncall,escalation,paging", """
This runbook describes how paging and escalation works when Beacon detects
an alert condition.

Sequence:
1. Beacon pages the primary on-call engineer for the owning team via
   Lighthouse.
2. If the page is not acknowledged within 10 minutes, Beacon automatically
   escalates to the secondary on-call engineer.
3. If there is still no acknowledgment 20 minutes after the original page,
   Beacon escalates to the team lead directly.
4. Acknowledging a page in the Beacon app stops further escalation but does
   not resolve the underlying alert; the engineer must still address or
   silence it.

Escalation timing is the same across all teams and is not currently
configurable per team.
"""),
    ("runbook-storage-cleanup", "Runbook: Ferry Storage Cleanup", "runbook", "ferry,storage,cleanup", """
Use this runbook when a Ferry bucket is approaching its storage quota.

Steps:
1. Check whether the bucket's cold-tier migration job is running
   successfully; a stalled migration job is the most common cause of
   unexpected quota pressure.
2. If migration is healthy but the bucket is still near quota, identify the
   largest and oldest objects as cold-storage migration candidates.
3. Deleting any object older than 30 days requires written approval from the
   bucket owner before it can be removed; this is a hard policy requirement,
   not a suggestion.
4. Once space is reclaimed, confirm the bucket has dropped below 80% of its
   quota before closing out the cleanup.
"""),
    ("api-aurora", "API Reference: Aurora Ingestion", "api_reference", "aurora,api,ingestion", """
POST /v1/events

Ingests a single event. Requires an Authorization header with a valid Anchor
access token. The request body must include `tenant_id`, `event_type`, and
a `payload` object.

Responses:
- 202 Accepted: event was buffered successfully.
- 413 Payload Too Large: payload exceeds the 256KB limit.
- 429 Too Many Requests: tenant exceeded its rate limit (default 5,000
  events/sec/tenant). The response includes a Retry-After header.

Successful responses include an `X-RateLimit-Remaining` header indicating
how much of the tenant's per-second budget is left.
"""),
    ("api-compass", "API Reference: Compass Gateway", "api_reference", "compass,api,gateway", """
Compass does not expose its own business-logic endpoints; it forwards
requests to backend services based on path prefix (for example, requests to
`/aurora/*` are forwarded to Aurora).

Compass adds an `X-Nimbus-Caller` header identifying the authenticated
caller before forwarding, and strips any client-supplied header with that
same name to prevent spoofing. All forwarded requests carry the gateway's
global rate limit headers as well as any service-specific limit headers
returned by the backend.
"""),
    ("api-anchor", "API Reference: Anchor Auth", "api_reference", "anchor,api,tokens", """
POST /v1/token — exchanges client credentials for a new access token (15
minute TTL) and refresh token (30 day TTL).

POST /v1/token/refresh — exchanges a valid refresh token for a new access
token without re-authenticating.

POST /v1/token/introspect — used internally by Compass to validate a
caller's access token; returns whether the token is active and its
associated scopes.

Access tokens are not revocable before expiry; because the TTL is only 15
minutes, Anchor relies on natural expiry rather than a revocation list.
"""),
    ("api-lighthouse", "API Reference: Lighthouse Notify", "api_reference", "lighthouse,api,notifications", """
POST /v1/notify

Sends a notification. Required fields: `channel` (one of `email`, `sms`,
`push`), `template`, and `recipient`. Optional `template_vars` object fills
placeholders in the chosen template.

Lighthouse returns a `notification_id` immediately and delivers
asynchronously. Callers that need delivery confirmation should register a
webhook URL, which Lighthouse calls with the final delivery status
(delivered, failed, or dead-lettered after 3 retries).
"""),
    ("api-ferry", "API Reference: Ferry Storage", "api_reference", "ferry,api,storage", """
PUT /v1/objects/{bucket}/{key} — uploads an object.
GET /v1/objects/{bucket}/{key} — downloads an object.
DELETE /v1/objects/{bucket}/{key} — deletes an object (subject to the
storage cleanup policy for objects older than 30 days).

GET /v1/objects/{bucket}/{key}?presign=true returns a time-limited presigned
URL instead of the object body, useful for letting a client download
directly without proxying through the caller.

Adding `?version={version_id}` to any read request retrieves a specific past
version of an object, if versioning is enabled on the bucket.
"""),
    ("postmortem-2026-03-aurora-outage", "Postmortem: Aurora Ring Buffer Overflow (March 2026)", "postmortem", "aurora,incident,postmortem", """
On March 2026, Aurora's ring buffer overflowed during a sudden traffic spike
from a single tenant, causing ingestion latency to climb to roughly 40
seconds for all tenants sharing that buffer shard.

Root cause: a recent deploy had introduced a second code path for event
ingestion that did not enforce the per-tenant rate cap, allowing the tenant
to exceed the normal 5,000 events/sec/tenant limit undetected.

Resolution: the missing rate cap was re-enabled on the affected path, and a
new Beacon alert was added to page on-call whenever ring buffer occupancy
exceeds 80%, rather than waiting for customer-visible latency to appear.

Follow-up action: all new ingestion code paths must now go through a shared
rate-limiting middleware instead of implementing their own checks.
"""),
    ("postmortem-2026-05-anchor-token", "Postmortem: Anchor Token TTL Misconfiguration (May 2026)", "postmortem", "anchor,incident,postmortem", """
On May 2026, a configuration typo caused Anchor to issue access tokens with
a 24 hour TTL instead of the intended 15 minutes.

The issue was caught within 40 minutes by an automated config-drift check
and reverted immediately. A security audit of tokens issued during the
window found no evidence of misuse.

Follow-up action: the config-drift check that caught this was previously
advisory-only (logged but did not page); it now pages the Nimbus-Platform
on-call engineer immediately when token TTL configuration drifts from the
approved value.
"""),
    ("postmortem-2026-06-lighthouse-delay", "Postmortem: Lighthouse Delivery Delay (June 2026)", "postmortem", "lighthouse,incident,postmortem", """
On June 2026, an outage at Lighthouse's SMS delivery provider caused SMS
notifications, including on-call pages, to be delayed by up to 25 minutes.

Lighthouse's automatic retry logic kept attempting delivery through the
failing provider rather than switching providers, since at the time
Lighthouse only supported a single SMS provider per environment.

Resolution: delivery was manually failed over to a backup SMS provider,
and the dead-letter queue was drained once the primary provider recovered.

Follow-up action: Lighthouse now supports an automatic provider failover
after 2 consecutive delivery failures on the same channel, rather than
requiring a manual switch.
"""),
    ("postmortem-2026-07-ferry-quota", "Postmortem: Ferry Storage Quota Exhaustion (July 2026)", "postmortem", "ferry,incident,postmortem", """
On July 2026, a Ferry bucket used for Cascade's transformed output hit its
storage quota, causing new writes from Cascade to fail for roughly 90
minutes.

Root cause: the bucket's cold-tier migration job had been silently failing
for about two weeks due to an expired internal credential, so objects that
should have moved to cheaper cold storage remained in the hot tier,
consuming quota far faster than expected.

Resolution: the credential was rotated, the migration job resumed, and the
bucket's quota was temporarily raised while cold-tier migration caught up.

Follow-up action: a new health check now alerts if the cold-tier migration
job has not completed successfully within 24 hours.
"""),
    ("policy-oncall-rotation", "Policy: On-Call Rotation", "policy", "oncall,policy,rotation", """
Each team maintains its own weekly on-call rotation with a primary and
secondary engineer. Rotation handoff happens every Monday at 10am, at which
point the outgoing primary briefs the incoming primary on any open issues.

Engineers who are paged outside normal business hours (before 9am or after
6pm local time, or on weekends) accrue comp time equal to the time spent
actively handling the page, not simply the time the page was open.
"""),
    ("policy-deploy-approval", "Policy: Deployment Approval", "policy", "deploy,policy,approval", """
All production deployments require at least 1 peer approval before they can
proceed, regardless of service or team.

Database migrations are held to a stricter standard: they require 2
approvals, and at least one of those approvals must come from a member of
the data team (Nimbus-Data), regardless of which team owns the service being
migrated.

Deploying on Friday afternoon is discouraged but not blocked; teams that do
so are expected to have an engineer available through the following morning.
"""),
    ("policy-data-retention", "Policy: Data Retention", "policy", "data,retention,policy", """
Raw event data ingested by Aurora is retained for 30 days before automatic
deletion. Aggregated and transformed data written by Cascade to Ferry is
retained for 1 year by default, configurable per bucket.

Postmortem documents are retained indefinitely and are never subject to
automatic deletion, regardless of the age or severity of the incident they
describe.
"""),
    ("policy-access-control", "Policy: Access Control", "policy", "access,security,policy", """
Access to production systems requires both multi-factor authentication and
an active VPN connection; neither alone is sufficient.

"Break-glass" emergency access (used when normal access paths are
unavailable during an incident) is logged automatically and reviewed by a
security lead every week, regardless of whether it was actually used for an
emergency.

All service-to-service authentication must use Anchor-issued tokens; direct
service-to-service credentials (shared secrets, static API keys) are not
permitted for new services.
"""),
    ("faq-rate-limits", "FAQ: Rate Limits", "faq", "rate-limits,faq", """
Q: What are the default rate limits across Nimbus services?
A: Aurora defaults to 5,000 events per second per tenant. Compass, as the
shared gateway, defaults to a global limit of 10,000 requests per second
across all traffic passing through it.

Q: How do I request a higher limit for my tenant?
A: Submit a change request describing the expected traffic and duration;
temporary increases (for example, for a product launch) are generally
approved faster than permanent increases.
"""),
    ("faq-new-service-onboarding", "FAQ: Onboarding a New Service", "faq", "onboarding,faq", """
Q: What do I need to do to onboard a new internal service onto the Nimbus
platform?
A: Three things: register the service and its routing rules with Compass,
request client credentials from Anchor so the service can authenticate, and
set up basic Beacon alert rules (at minimum, an availability check) before
the service is considered production-ready.

Q: Is there a required order for these steps?
A: Register with Compass and Anchor first; Beacon alerting can be set up
once the service is actually deployed and emitting metrics.
"""),
    ("faq-token-expiry", "FAQ: Why Did My Token Stop Working", "faq", "anchor,tokens,faq", """
Q: My access token stopped working after about 15 minutes. Is that a bug?
A: No, this is expected. Anchor issues access tokens with a 15 minute TTL by
design. Use your refresh token (valid for 30 days) to obtain a new access
token instead of re-authenticating from scratch.

Q: What happens when my refresh token itself expires after 30 days?
A: You must re-authenticate with Anchor from scratch to obtain a new
refresh token and access token pair.
"""),
    ("faq-alert-noise", "FAQ: Reducing Alert Noise", "faq", "beacon,alerting,faq", """
Q: Beacon is paging our team too often for the same underlying issue. How do
we reduce this?
A: Three common options: raise the alert threshold if it's too sensitive,
add an inhibition rule so a root-cause alert suppresses related downstream
alerts, or configure a maintenance window to silence alerts during planned
work.

Q: Does silencing an alert during a maintenance window affect other teams?
A: No, maintenance windows are scoped to the specific service and alert
rule you configure them for.
"""),
    ("faq-support-contacts", "FAQ: Who Owns What", "faq", "contacts,faq,teams", """
Q: Which team do I contact for issues with a given service?
A: Nimbus-Core owns Aurora and Compass. Nimbus-Platform owns Anchor, Beacon,
and Lighthouse. Nimbus-Data owns Cascade and Ferry.

Q: Is there a single channel for cross-team platform questions?
A: Yes, general platform questions that don't clearly belong to one team can
go to the shared platform-questions channel, which is triaged by whichever
team is best suited to answer.
"""),
    ("faq-slo-targets", "FAQ: SLO Targets by Tier", "faq", "slo,tiers,faq", """
Q: What uptime SLO applies to a given service?
A: It depends on the service's tier. Tier 1 services target 99.9% uptime,
which allows roughly 43 minutes of downtime per month. Tier 2 services
target 99.5%, roughly 3.6 hours per month. Tier 3 services target 99%,
roughly 7.3 hours per month.

Q: How is a service's tier decided?
A: Tier is assigned based on how directly customer-facing and
business-critical the service is; Aurora, Compass, and Anchor are Tier 1
because most other services depend on them.
""",
    ),
]

# ---------------------------------------------------------------------------
# Structured records: services + teams (data/records.json)
# ---------------------------------------------------------------------------

RECORDS = {
    "services": [
        {"id": "aurora", "type": "service", "name": "Aurora", "description": "Event ingestion service",
         "owner_team": "nimbus-core", "tier": 1, "uptime_slo_pct": 99.9, "region": "us-east-1", "last_deploy": "2026-08-12"},
        {"id": "compass", "type": "service", "name": "Compass", "description": "API gateway",
         "owner_team": "nimbus-core", "tier": 1, "uptime_slo_pct": 99.9, "region": "us-east-1", "last_deploy": "2026-08-20"},
        {"id": "anchor", "type": "service", "name": "Anchor", "description": "Authentication and identity service",
         "owner_team": "nimbus-platform", "tier": 1, "uptime_slo_pct": 99.9, "region": "us-east-1", "last_deploy": "2026-08-05"},
        {"id": "beacon", "type": "service", "name": "Beacon", "description": "Monitoring and alerting service",
         "owner_team": "nimbus-platform", "tier": 2, "uptime_slo_pct": 99.5, "region": "us-east-1", "last_deploy": "2026-07-29"},
        {"id": "lighthouse", "type": "service", "name": "Lighthouse", "description": "Notification delivery service",
         "owner_team": "nimbus-platform", "tier": 2, "uptime_slo_pct": 99.5, "region": "us-east-1", "last_deploy": "2026-08-15"},
        {"id": "cascade", "type": "service", "name": "Cascade", "description": "Batch data pipeline / ETL",
         "owner_team": "nimbus-data", "tier": 2, "uptime_slo_pct": 99.5, "region": "us-west-2", "last_deploy": "2026-08-01"},
        {"id": "ferry", "type": "service", "name": "Ferry", "description": "Object storage service",
         "owner_team": "nimbus-data", "tier": 2, "uptime_slo_pct": 99.5, "region": "us-west-2", "last_deploy": "2026-07-22"},
    ],
    "teams": [
        {"id": "nimbus-core", "type": "team", "name": "Nimbus-Core", "slack_channel": "#nimbus-core",
         "oncall_primary": "j.alvarez", "oncall_secondary": "s.chen"},
        {"id": "nimbus-platform", "type": "team", "name": "Nimbus-Platform", "slack_channel": "#nimbus-platform",
         "oncall_primary": "r.osei", "oncall_secondary": "m.kaur"},
        {"id": "nimbus-data", "type": "team", "name": "Nimbus-Data", "slack_channel": "#nimbus-data",
         "oncall_primary": "t.nguyen", "oncall_secondary": "d.ibrahim"},
    ],
}

# ---------------------------------------------------------------------------
# Evaluation benchmark (data/benchmark.json)
# ---------------------------------------------------------------------------

BENCHMARK = [
    {"id": "q01", "question": "What is the default per-tenant rate limit for Aurora event ingestion?",
     "category": "pure_document", "difficulty": "easy",
     "expected_doc_ids": ["arch-aurora", "api-aurora", "faq-rate-limits"],
     "expected_answer": "5,000 events per second per tenant", "expected_tool": None},
    {"id": "q02", "question": "How long is an Anchor access token valid before it expires?",
     "category": "pure_document", "difficulty": "easy",
     "expected_doc_ids": ["arch-anchor", "api-anchor", "faq-token-expiry"],
     "expected_answer": "15 minutes", "expected_tool": None},
    {"id": "q03", "question": "What caused the March 2026 Aurora outage and how was it fixed?",
     "category": "pure_document", "difficulty": "medium",
     "expected_doc_ids": ["postmortem-2026-03-aurora-outage"],
     "expected_answer": "A deploy introduced a second ingestion path that skipped the per-tenant rate cap, causing a ring buffer overflow; fixed by re-enabling the cap and adding a buffer-occupancy alert.",
     "expected_tool": None},
    {"id": "q04", "question": "What are the RTO and RPO for a Cascade database failover?",
     "category": "pure_document", "difficulty": "medium",
     "expected_doc_ids": ["runbook-db-failover"],
     "expected_answer": "RTO of 5 minutes and RPO of 30 seconds", "expected_tool": None},
    {"id": "q05", "question": "How many times does Lighthouse retry a failed notification before giving up?",
     "category": "pure_document", "difficulty": "easy",
     "expected_doc_ids": ["arch-lighthouse", "api-lighthouse"],
     "expected_answer": "3 retries with exponential backoff, then it is dead-lettered", "expected_tool": None},
    {"id": "q06", "question": "According to policy, how many approvals does a production database migration need, and from whom?",
     "category": "pure_document", "difficulty": "hard",
     "expected_doc_ids": ["policy-deploy-approval"],
     "expected_answer": "2 approvals, at least one from the data team (Nimbus-Data)", "expected_tool": None},
    {"id": "q07", "question": "Which team owns the Ferry service?",
     "category": "structured_record", "difficulty": "easy",
     "expected_doc_ids": [], "expected_record_ids": ["ferry"],
     "expected_answer": "Nimbus-Data", "expected_tool": "lookup_record"},
    {"id": "q08", "question": "What tier is the Compass service, and what uptime SLO does that imply?",
     "category": "structured_record", "difficulty": "medium",
     "expected_doc_ids": ["faq-slo-targets"], "expected_record_ids": ["compass"],
     "expected_answer": "Tier 1, 99.9% uptime SLO", "expected_tool": "lookup_record"},
    {"id": "q09", "question": "What is Beacon's uptime SLO percentage?",
     "category": "structured_record", "difficulty": "easy",
     "expected_doc_ids": [], "expected_record_ids": ["beacon"],
     "expected_answer": "99.5%", "expected_tool": "lookup_record"},
    {"id": "q10", "question": "A Tier 1 service has an uptime SLO of 99.9%, which allows about 43 minutes of downtime per month. If Aurora has already used 17 minutes this month, how many minutes of downtime budget remain?",
     "category": "tool_arithmetic", "difficulty": "easy",
     "expected_doc_ids": ["faq-slo-targets"], "expected_record_ids": [],
     "expected_answer": "26 minutes", "expected_tool": "calculator"},
    {"id": "q11", "question": "Beacon scrapes metrics every 30 seconds. How many scrapes happen in one hour?",
     "category": "tool_arithmetic", "difficulty": "easy",
     "expected_doc_ids": ["arch-beacon"], "expected_record_ids": [],
     "expected_answer": "120", "expected_tool": "calculator"},
    {"id": "q12", "question": "Who should be paged if the team that owns Cascade needs to be escalated to, and how long after the first page does escalation to secondary happen?",
     "category": "multi_hop", "difficulty": "medium",
     "expected_doc_ids": ["runbook-oncall-escalation"], "expected_record_ids": ["cascade"],
     "expected_answer": "Nimbus-Data's on-call; escalates to secondary after 10 minutes without acknowledgment",
     "expected_tool": "lookup_record"},
    {"id": "q13", "question": "If Ferry's July 2026 storage quota issue happened again, which team owns the fix, and what does policy require before deleting anything older than 30 days?",
     "category": "multi_hop", "difficulty": "hard",
     "expected_doc_ids": ["postmortem-2026-07-ferry-quota", "runbook-storage-cleanup"], "expected_record_ids": ["ferry"],
     "expected_answer": "Nimbus-Data; deleting objects older than 30 days requires written approval from the bucket owner",
     "expected_tool": "lookup_record"},
    {"id": "q14", "question": "Aurora's rate limit is 5,000 events per second per tenant. Roughly how many events could one tenant send in a 10 minute window at that limit?",
     "category": "multi_hop", "difficulty": "hard",
     "expected_doc_ids": ["arch-aurora"], "expected_record_ids": [],
     "expected_answer": "3,000,000 events", "expected_tool": "calculator"},
    {"id": "q15", "question": "What was Nimbus's total revenue last quarter?",
     "category": "out_of_corpus", "difficulty": "medium",
     "expected_doc_ids": [], "expected_record_ids": [],
     "expected_answer": "not answerable from the knowledge base", "expected_tool": None},
    {"id": "q16", "question": "Who is the CEO of the company that runs Nimbus?",
     "category": "out_of_corpus", "difficulty": "medium",
     "expected_doc_ids": [], "expected_record_ids": [],
     "expected_answer": "not answerable from the knowledge base", "expected_tool": None},
    {"id": "q17", "question": "What programming language is Aurora written in?",
     "category": "out_of_corpus", "difficulty": "easy",
     "expected_doc_ids": [], "expected_record_ids": [],
     "expected_answer": "not answerable from the knowledge base", "expected_tool": None},
    {"id": "q18", "question": "List all services owned by Nimbus-Platform.",
     "category": "structured_record", "difficulty": "medium",
     "expected_doc_ids": [], "expected_record_ids": ["anchor", "beacon", "lighthouse"],
     "expected_answer": "Anchor, Beacon, and Lighthouse", "expected_tool": "search_records"},
    {"id": "q19", "question": "What channels can Lighthouse use to deliver a notification?",
     "category": "pure_document", "difficulty": "easy",
     "expected_doc_ids": ["arch-lighthouse", "api-lighthouse"],
     "expected_answer": "email, SMS, and push", "expected_tool": None},
    {"id": "q20", "question": "Compass has a default global rate limit of 10,000 requests per second. If that traffic were split evenly across 7 backend services, roughly how many requests per second would each service receive?",
     "category": "tool_arithmetic", "difficulty": "hard",
     "expected_doc_ids": ["arch-compass"], "expected_record_ids": [],
     "expected_answer": "about 1,428.6 requests per second", "expected_tool": "calculator"},
]


def write_documents() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, title, category, tags, body in DOCS:
        frontmatter = (
            f"---\n"
            f"doc_id: {doc_id}\n"
            f"title: {title}\n"
            f"category: {category}\n"
            f"tags: {tags}\n"
            f"---\n"
        )
        (DOCS_DIR / f"{doc_id}.md").write_text(frontmatter + body.strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(DOCS)} documents to {DOCS_DIR}")


def write_records() -> None:
    path = ROOT / "data" / "records.json"
    path.write_text(json.dumps(RECORDS, indent=2), encoding="utf-8")
    print(f"Wrote records to {path}")


def write_benchmark() -> None:
    path = ROOT / "data" / "benchmark.json"
    path.write_text(json.dumps(BENCHMARK, indent=2), encoding="utf-8")
    print(f"Wrote {len(BENCHMARK)} benchmark queries to {path}")


if __name__ == "__main__":
    write_documents()
    write_records()
    write_benchmark()
