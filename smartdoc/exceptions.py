"""Domain exceptions with user-facing messages."""

from __future__ import annotations


class SmartDocError(Exception):
    """Base error for SmartDoc AI."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class UnsupportedFileTypeError(SmartDocError):
    pass


class EmptyDocumentError(SmartDocError):
    pass


class CorruptedFileError(SmartDocError):
    pass


class ExtractionError(SmartDocError):
    pass


class FileTooLargeError(SmartDocError):
    pass


class ConfigurationError(SmartDocError):
    pass


class MissingAPIKeyError(ConfigurationError):
    pass


class EmbeddingError(SmartDocError):
    pass


class VectorStoreError(SmartDocError):
    pass


class RetrievalError(SmartDocError):
    pass


class LLMError(SmartDocError):
    pass


class InvalidQuestionError(SmartDocError):
    pass


class NoRelevantContextError(SmartDocError):
    pass
