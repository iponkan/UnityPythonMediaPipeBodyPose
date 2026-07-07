#pipe server
from body import BodyThread
import time
import global_vars
import signal


def request_exit(signum=None, frame=None):
    print("Exiting...")
    global_vars.KILL_THREADS = True


thread = BodyThread()
thread.daemon = True

signal.signal(signal.SIGINT, request_exit)
thread.start()

try:
    while thread.is_alive() and not global_vars.KILL_THREADS:
        thread.join(0.2)
except KeyboardInterrupt:
    request_exit()

thread.join(2.0)
