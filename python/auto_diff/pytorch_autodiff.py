
import torch



def main():
    x, a = torch.zeros(3, requires_grad=True), torch.rand(3)
    c = torch.dot(a, x)
    c.backward()
    print(f"Gradient Should be {a}")
    print(x.grad)
    return None



if __name__ == "__main__": 
    print(f"Script running: {__file__}")
    main()