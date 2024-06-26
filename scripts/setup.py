import os
import subprocess
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def run_command(command):
    """Run a shell command and print its output."""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr.decode())
    else:
        print(result.stdout.decode())

# Step 1: Clone the repositories if they don't already exist
print("Checking for repositories...")

if not os.path.isdir("GroundingDINO"):
    print("Cloning GroundingDINO repository...")
    run_command("git clone https://github.com/IDEA-Research/GroundingDINO.git")
else:
    print("GroundingDINO repository already exists.")

if not os.path.isdir("segment-anything"):
    print("Cloning Segment Anything repository...")
    run_command("git clone https://github.com/facebookresearch/segment-anything.git")
else:
    print("Segment Anything repository already exists.")

# Step 2: Install the packages
print("Installing packages from GroundingDINO...")
run_command("cd GroundingDINO && pip install -q -e .")

# Step 3: Create weights directory and download the weights if they don't already exist
print("Checking for weights...")

if not os.path.isfile("weights/groundingdino_swint_ogc.pth"):
    print("Downloading GroundingDINO weights...")
    os.makedirs('weights', exist_ok=True)
    run_command("wget -q -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
else:
    print("GroundingDINO weights already exist.")

if not os.path.isfile("weights/sam_vit_h_4b8939.pth"):
    print("Downloading SAM weights...")
    os.makedirs('weights', exist_ok=True)
    run_command("wget -q -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
else:
    print("SAM weights already exist.")

print("Setup complete!")

# Paths and constants
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
