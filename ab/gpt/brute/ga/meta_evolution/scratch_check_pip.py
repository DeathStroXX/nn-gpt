import subprocess

result = subprocess.run(
    ["kubectl", "logs", "job/nngpt-fractal-meta-evo-clone3-cifar10"], 
    capture_output=True, text=True
)

lines = result.stdout.split('\n')
for i, line in enumerate(lines):
    if "Successfully installed" in line:
        print("\n".join(lines[max(0, i-5):i+5]))
        break
