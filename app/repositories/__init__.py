"""Repository layer: persistence access for runs and their results.

Isolates SQLite/storage access behind a narrow interface so services and the
worker don't embed SQL. Backed by the engine's SQLite schema (report_cleanup.db).
"""
