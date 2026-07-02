"""Service layer: orchestrates the engine, storage, and queue.

Routers stay thin and delegate here; services are the only layer that talks to
the `report_cleanup` engine, OneDrive, the queue, and the repositories.
"""
