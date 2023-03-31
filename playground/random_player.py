import torch

def main():
    torch.initial_seed()
    print(torch.randint(100, size = (1,1)).item())

if __name__ == "__main__":
    main()