import torch
import torchvision
import torchaudio

def verify_installation():
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    print("Torchaudio Version:", torchaudio.__version__)
    print("\nCUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("Current CUDA Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

if __name__ == "__main__":
    verify_installation() 