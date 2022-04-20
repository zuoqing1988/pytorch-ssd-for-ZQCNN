from .transforms import *


class TrainAugmentation:
    def __init__(self, size_x, size_y, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size_x = size_x
        self.size_y = size_y
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(self.size_x,self.size_y),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size_x, self.size_y),
            #SubtractMeans(self.mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        #print(boxes)
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size_x, size_y, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size_x, size_y),
            #SubtractMeans(mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size_x, size_y, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size_x, size_y),
            #SubtractMeans(mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
