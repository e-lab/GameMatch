import torch
import torch.nn as nn

from torchvision import models
from glob import glob

class Detector:
    def __init__(self):
        self.features = models.alexnet(pretrained=True).features
        self.last = None
        # self.distfunc = nn.PairwiseDistance(p=2)
        self.distfunc = nn.CosineSimilarity()
        self.threshold = 1.0
        
    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), 256*6*6)

        print(self.last)

        if self.last is None:
            self.last = x.clone()
            return x

        distance = self.distfunc(x, self.last)
        print(distance)

        self.last = x.clone()

        if distance > self.threshold:
            return x
        else:
            return None


def test():
    from PIL import Image as img
    from torch.autograd import Variable
    from torchvision import transforms
    # these are very similar images
    lof = sorted(glob('data_2room/screens/*.pth'))
    lof = lof[:500]
    #print(lof)
    
    d1 = torch.load(lof[3])
    d2 = torch.load(lof[4])

    d4 = torch.load(lof[349])

    im1 = img.fromarray(d1)
    im2 = img.fromarray(d2)
    im4 = img.fromarray(d4)

    '''
    im1.show(title='1')
    im2.show(title='2')
    im4.show(title='4')
    '''

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
            transforms.Resize(256),  # transforms.Scale(256),
            transforms.CenterCrop(224)# ,
            # transforms.ToTensor(),
            # normalize
    ])
    # d1, d2, d4 = transformations(im1), transformations(im2), transformations(im4)
    im1, im2, im4 = transformations(im1), transformations(im2), transformations(im4)
    # print(d1, d2, d4)

    '''
    im1 = img.fromarray(d1)
    im2 = img.fromarray(d2)
    im4 = img.fromarray(d4)
    '''

    im1.show(title='1')
    im2.show(title='2')
    im4.show(title='4')

    d1, d2, d4 = Variable(d1), Variable(d2), Variable(d4)
    # d1, d2, d4 = Variable(transformations(im1).unsqueeze_(0)), Variable(transformations(im2).unsqueeze_(0)), Variable(transformations(im4).unsqueeze_(0))


    detector = Detector()
    detector.forward(d1)
    detector.forward(d4)
    
    detector1 = Detector()
    detector1.forward(d1)
    detector1.forward(d2)


if __name__ == '__main__':
    test()

