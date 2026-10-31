"""Audit logger: an append-only JSONL record of every agent decision and tool call.

Named audit.py (not logging.py) so it never shadows the stdlib logging module.

TODO: implement log_event(kind, payload) writing to <GOTRAIN_HOME>/logs/.
"""
