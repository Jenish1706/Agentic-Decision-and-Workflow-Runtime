---
doc_id: policy-deploy-approval
title: Policy: Deployment Approval
category: policy
tags: deploy,policy,approval
---
All production deployments require at least 1 peer approval before they can
proceed, regardless of service or team.

Database migrations are held to a stricter standard: they require 2
approvals, and at least one of those approvals must come from a member of
the data team (Nimbus-Data), regardless of which team owns the service being
migrated.

Deploying on Friday afternoon is discouraged but not blocked; teams that do
so are expected to have an engineer available through the following morning.
