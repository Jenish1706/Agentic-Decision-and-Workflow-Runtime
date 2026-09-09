---
doc_id: runbook-oncall-escalation
title: Runbook: On-Call Escalation
category: runbook
tags: oncall,escalation,paging
---
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
