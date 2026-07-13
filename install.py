import subprocess
import sys

def has_nvidia_gpu():
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def main():
    if has_nvidia_gpu():
        print("NVIDIA GPU detected. Installing CUDA PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0", "torchvision", "torchaudio"])
    else:
        print("No NVIDIA GPU detected. Installing CPU-only PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    
    print("Installing remaining dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    main()
