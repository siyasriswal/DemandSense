"""
Starts the whole application: the backend API and the dashboard.

    python start.py

Backend API  : http://127.0.0.1:8000   (interactive docs at /docs)
Dashboard    : http://localhost:8501
Press Ctrl+C once to stop both.
"""
import subprocess
import sys
import time
import urllib.request
import webbrowser

from config import API_HOST, API_PORT


def wait_for_api(timeout=40):
    url = f"http://{API_HOST}:{API_PORT}/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    py = sys.executable
    api = subprocess.Popen([py, "-m", "uvicorn", "backend.main:app", "--host", API_HOST, "--port", str(API_PORT)])
    if not wait_for_api():
        api.terminate()
        sys.exit("The backend did not start. Run `python -m uvicorn backend.main:app` to see the error.")
    print(f"Backend ready on port {API_PORT}. Starting dashboard...")
    ui = subprocess.Popen([py, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    try:
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()


if __name__ == "__main__":
    main()
