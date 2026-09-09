---
doc_id: postmortem-2026-07-ferry-quota
title: Postmortem: Ferry Storage Quota Exhaustion (July 2026)
category: postmortem
tags: ferry,incident,postmortem
---
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
