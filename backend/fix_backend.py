import os
import subprocess
import time

print("🔍 Searching for any hidden processes using port 8000...")
try:
    # Find the process id using port 8000
    output = subprocess.check_output("netstat -ano | findstr :8000", shell=True).decode()
    time.sleep(1)
    for line in output.splitlines():
        if 'LISTENING' in line:
            pid = line.strip().split()[-1]
            print(f"💀 Found process {pid} using port 8000. Terminating it...")
            os.system(f"taskkill /F /PID {pid}")
            print("✅ Port 8000 is now free!")
            break
except Exception as e:
    print("No process found blocking port 8000, or an error occurred.")


print("\n🚀 You can now start your backend!")
print("Please run: ..\\start_backend.bat")
