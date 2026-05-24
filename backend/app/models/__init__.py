"""Backend inference modules.

Heavy PyTorch training classes remain available from ``app.models.emotion_model``
when explicitly needed. Avoid importing them here so the lightweight API
inference path can start without eagerly loading training dependencies.
"""
