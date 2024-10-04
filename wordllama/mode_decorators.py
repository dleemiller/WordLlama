from functools import wraps


def dense_only(method):
    """Decorator to ensure the method is only called when using dense embeddings."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.binary:
            raise ValueError(
                f"The method '{method.__name__}' is only implemented for dense embeddings."
            )
        return method(self, *args, **kwargs)

    return wrapper


def binary_only(method):
    """Decorator to ensure the method is only called when using binary embeddings."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.binary:
            raise ValueError(
                f"The method '{method.__name__}' is only implemented for binary embeddings."
            )
        return method(self, *args, **kwargs)

    return wrapper
