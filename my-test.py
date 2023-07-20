import torch
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':
    dilation = 0
    model = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=dilation,
                      groups=1, bias=False, dilation=dilation)
    input = Variable(torch.FloatTensor(2,3,256,256))
    output = model(input)
    print(output.size())