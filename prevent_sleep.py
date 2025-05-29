import ctypes
import time

# Windows API flags
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """
    Inhibit system sleep and display turn-off.
    Call this periodically to keep the system awake.
    """
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS |
        ES_SYSTEM_REQUIRED |
        ES_DISPLAY_REQUIRED
    )

def restore_sleep():
    """
    Clear the execution state flags to restore normal behavior.
    """
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    try:
        print("Preventing sleep and display timeout. Press Ctrl+C to stop.")
        # Loop indefinitely, refreshing every 30 seconds
        while True:
            prevent_sleep()
            time.sleep(30)
    except KeyboardInterrupt:
        restore_sleep()
        print("Normal sleep behavior restored.")
