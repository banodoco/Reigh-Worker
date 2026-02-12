"""
Platform-specific utilities for headless operation.

This module contains utilities for handling platform-specific quirks
when running in headless environments (no display, no audio, etc.).
"""


def suppress_alsa_errors():
    """
    Suppress ALSA error messages on Linux.

    ALSA (Advanced Linux Sound Architecture) prints verbose error messages
    to stderr when no audio device is available. This is common in headless
    GPU workers. This function installs a null error handler to silence them.

    Safe to call on non-Linux systems - does nothing if ALSA is unavailable.
    """
    try:
        import ctypes

        # Define the ALSA error handler function type
        # void (*)(const char *file, int line, const char *function, int err, const char *fmt)
        handler_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,  # file
            ctypes.c_int,     # line
            ctypes.c_char_p,  # function
            ctypes.c_int,     # err
            ctypes.c_char_p   # fmt
        )

        # Create a null handler that does nothing
        def null_handler(filename, line, function, err, fmt):
            pass

        c_null_handler = handler_type(null_handler)

        # Load ALSA library and set the error handler
        libasound = ctypes.cdll.LoadLibrary('libasound.so.2')
        libasound.snd_lib_error_set_handler(c_null_handler)

        # Keep a reference to prevent garbage collection
        suppress_alsa_errors._handler = c_null_handler

    except OSError:
        pass  # ALSA not available (not on Linux or libasound not installed)
    except AttributeError:
        pass  # ctypes not fully available
    except ImportError:
        pass  # ctypes module not available


def setup_headless_environment():
    """
    Set up environment variables for headless operation.

    Call this early in application startup before importing libraries
    that may check for display/audio availability.
    """
    import os

    # Suppress Python warnings
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

    # Set XDG runtime directory (required by some libraries)
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")

    # Use dummy audio driver for SDL/pygame
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # Hide pygame support prompt
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    # Suppress ALSA errors
    suppress_alsa_errors()
