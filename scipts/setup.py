import os
import subprocess

def run_command(command):
    """Run a shell command and print its output."""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr.decode())
    else:
        print(result.stdout.decode())

# Step 1: Clone the repositories
print("Cloning repositories...")
run_command("git clone https://github.com/IDEA-Research/GroundingDINO.git")
run_command("git clone https://github.com/facebookresearch/segment-anything.git")

# Step 2: Install the packages
print("Installing packages from GroundingDINO...")
run_command("cd GroundingDINO && pip install -q -e .")

# Step 3: Create weights directory and download the weights
print("Creating weights directory and downloading weights...")
os.makedirs('weights', exist_ok=True)
run_command("wget -q -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
run_command("wget -q -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

print("Setup complete!")
