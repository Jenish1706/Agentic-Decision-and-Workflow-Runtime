---
doc_id: runbook-deploy-rollback
title: Runbook: Deployment Rollback
category: runbook
tags: deploy,rollback,operations
---
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
