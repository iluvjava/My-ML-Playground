import torch
import numpy

def main():
    x = torch.rand((10, 1), requires_grad=True)
    xViewed = x.view((-1, ))
    def compute(x):
        return (1/2)*torch.dot(x, x)
    y = compute(xViewed)
    y.backward()
    with torch.no_grad():
        # need to do it in this context because numpy can't compute with grad. 
        assert numpy.isclose(x.grad, x).all()
        print("Seems like we got it correct")
    return None


if __name__ == "__main__": 
    print(f"Script running: {__file__}")
    main()