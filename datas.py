import torch.utils.data as data
from torch.autograd import Variable

from data import v2, UCF24Detection, AnnotationTransform, detection_collate, CLASSES_JHMDB, BaseTransform
from utils.augmentations_ import SSDAugmentation


train_dataset = UCF24Detection("/mnt/data/Action/data/ucf24/ucf24/", 'train', SSDAugmentation(300, (104, 117, 123)),
                               AnnotationTransform(), input_type='rgb')
train_data_loader = data.DataLoader(train_dataset, 1, num_workers=1,
                                    shuffle=True, collate_fn=detection_collate, pin_memory=True)
for i, (images, targets, img_indexs) in enumerate(train_data_loader):
    images = Variable(images)
    targets = [Variable(anno, volatile=True) for anno in targets]