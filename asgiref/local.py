import contextlib
import contextvars
import threading


class _CVar:
    """Storage utility for Local."""

    def __init__(self):
        self._data = contextvars.ContextVar("asgiref.local")

    def __getattr__(self, key):
        storage_object = self._data.get({})
        try:
            return storage_object[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value) -> None:
        if key == "_data":
            return super().__setattr__(key, value)

        storage_object = self._data.get({})
        storage_object[key] = value
        self._data.set(storage_object)

    def __delattr__(self, key) -> None:
        storage_object = self._data.get({})
        del storage_object[key]
        self._data.set(storage_object)


class Local:
    """Local storage for async tasks.

    This is a namespace object (similar to `threading.local`) where data is
    also local to the current async task.

    This is implemented using the `contextvars` module and so "local" means
    within a context as defined by the `contextvars` module. Generally data
    visible in the current scope will still be visible:

    - to coroutines directly awaited from the current scope.
    - to tasks spawned using `asyncio` utilities (`create_task`, `wait_for`,
      `gather` and probably others).
    - when explicitly maintaining the context using functions from the
      `contextvars` module like `context.run`.
    - anything spawned by `async_to_sync` or `sync_to_async`

    Data stored on this object in a given scope is NOT visible:

    - to async tasks that are running in "parallel" (not spawned or awaited
      from the current scope).
    - when the context has been intentionally changed for another one by e.g.
      created using `contextvars.copy_context`.
    - to code running in a new thread.

    If `thread_critical` is `False`, data can still be accessed from threads
    spawned from the current thread if the context is copied across - this
    happens when the thread is spawned by `async_to_sync`. If
    `thread_critical` is set to `True`, the data will always be thread-local
    and will not be transferred to new threads even when using `async_to_sync`.

    Unlike plain `contextvars` objects, this utility is threadsafe.
    """

    def __init__(self, thread_critical=False):
        self._thread_critical = thread_critical
        self._thread_lock = threading.RLock()

        if thread_critical:
            # Thread-local storage
            self._storage = threading.local()
        else:
            # Contextvar storage
            self._storage = _CVar()

    @contextlib.contextmanager
    def _lock_storage(self):
        # Thread safe access to storage
        if self._thread_critical:
            # Ensure context exists in the current thread
            if not hasattr(self._storage, "cvar"):
                self._storage.cvar = _CVar()

            # self._storage is a thread local, so the members
            # can't be accessed in another thread (we don't
            # need any locks)
            yield self._storage.cvar
        else:
            # Lock for thread_critical=False as other threads
            # can access the exact same storage object
            with self._thread_lock:
                yield self._storage

    def __getattr__(self, key):
        with self._lock_storage() as storage:
            return getattr(storage, key)

    def __setattr__(self, key, value):
        if key in ("_local", "_storage", "_thread_critical", "_thread_lock"):
            return super().__setattr__(key, value)
        with self._lock_storage() as storage:
            setattr(storage, key, value)

    def __delattr__(self, key):
        with self._lock_storage() as storage:
            delattr(storage, key)
