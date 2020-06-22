import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Function

__all__ = ['AlexNet', 'alexnet']

model_urls = {

    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',

}

class AlexNet(nn.Module):

  def __init__(self, num_classes=1000):

    super(AlexNet, self).__init__()

    self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    self.new_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

  def forward(self, x, Gd = None):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    if Gd:
      reverse_feature = ReverseLayerF.apply(x, Gd)
      x = self.new_classifier(reverse_feature)
    else:
      x = self.classifier(x)
    return x


def alexnet(pretrained=False, progress=True, **kwargs):

    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    net = AlexNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        net.load_state_dict(state_dict, strict = False)
        
        # Copy weights of the original classifier into the new classifier
         # Adjust weights
        net.new_classifier[1].weight.data = net.classifier[1].weight.data.clone()
        net.new_classifier[1].bias.data = net.classifier[1].bias.data.clone()

        net.new_classifier[4].weight.data = net.classifier[4].weight.data.clone()
        net.new_classifier[4].bias.data = net.classifier[4].bias.data.clone()
        
    return net

class ReverseLayerF(Function):

    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
