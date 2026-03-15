class ServiceBusyError(RuntimeError):
    """Raised when a service is already processing another request."""


class ServiceNotReadyError(RuntimeError):
    """Raised when a service has not completed startup preload."""


class InvalidInputError(ValueError):
    """Raised when request data is missing or inconsistent."""
