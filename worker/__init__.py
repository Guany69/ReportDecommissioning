"""Asynchronous processing worker.

Consumes analysis-run jobs enqueued by the API and executes the engine out of
band, so HTTP requests return immediately with a run id instead of blocking on a
long-running analysis.
"""
