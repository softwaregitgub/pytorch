import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

# model = models.alexnet(pretrained=True)
# backprop = Backprop(model)
#
# image = load_image('./dog.jpg')
# owl = apply_transforms(image)
#
# target_class = 24
# backprop.visualize(owl, target_class, guided=True, use_gpu=True)

import torchvision.models as models
from flashtorch.activmax import GradientAscent
model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)
# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]
g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")


#   https://andrewhuman.github.io/cnn-hidden-layout_search
# 【2】https://github.com/jacobgil/pytorch-grad-cam
# 【3】https://github.com/MisaOgura/flashtorch