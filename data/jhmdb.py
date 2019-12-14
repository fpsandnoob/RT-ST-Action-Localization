import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np

CLASSES_JHMDB = (  # always index 0
    'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
    'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball',
    'throw', 'walk', 'wave')


class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of UCF24's 24 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES_JHMDB, range(len(CLASSES_JHMDB))))
        self.ind_to_class = dict(zip(range(len(CLASSES_JHMDB)), CLASSES_JHMDB))

    def __call__(self, bboxs, labels, width, height):
        res = []
        for t in range(len(labels)):
            bbox = bboxs[t,:]
            label = labels[t]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            for i in range(4):
                cur_pt = max(0,int(bbox[i]) - 1)
                scale =  width if i % 2 == 0 else height
                cur_pt = min(scale, int(bbox[i]))
                cur_pt = float(cur_pt) / scale
                bndbox.append(cur_pt)
            bndbox.append(label)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        vid = vid.replace('.avi', '')
        trainvideos.append(vid)
    return trainvideos


def make_lists(rootpath, imgtype, split=1, fulltest=False):
    imagesDir = rootpath + imgtype + '/'
    splitfile = '/home/user/git/realtime-action-detection/data/trainlist{:02d}.txt'.format(split)
    testvideos = readsplitfile('/home/user/git/realtime-action-detection/data/testlist{:02d}.txt'.format(split))
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []

    with open('data/jhmdb/jhmdb_.pkl','rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES_JHMDB), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES_JHMDB), dtype=np.int32)

    # ratios = np.asarray([1.1, 0.8, 4.7, 1.4, 0.9, 2.6, 2.2, 3.0, 3.0, 5.0, 6.2, 2.7,
    #                     3.5, 3.1, 4.3, 2.5, 4.5, 3.4, 6.7, 3.6, 1.6, 3.4, 0.6, 4.3])
    ratios = np.asarray([0.893810904,1.155414096,0.981011968,0.872010639,0.937411436,0.91561117,
                         0.959211702,0.872010639, 1.220814894,1.199014628,0.937411436,0.959211702,
                         0.981011968, 1.155414096,1.199014628,0.872010639,0.893810904,1.177214362,
                         1.002812234, 0.893810904,0.91561117
                         ])
    # ratios = np.asarray([1.03, 0.75, 4.22, 1.32, 0.8, 2.36, 1.99, 2.66, 2.68, 4.51, 5.56, 2.46, 3.17, 2.76, 3.89, 2.28, 4.01, 3.08, 6.06, 3.28, 1.51, 3.05, 0.6, 3.84])
    # ratios = np.ones_like(ratios) #TODO:uncomment this line and line 155, 156 to compute new ratios might be useful for JHMDB21
    video_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']
        istrain = True
        step = ratios[actidx]
        numf = database[videoname]['numf']
        lastf = numf-1
        if videoname in trainvideos or videoname in testvideos:
            if videoname in testvideos:
                istrain = False
                step = max(1, ratios[actidx])*3
            if fulltest:
                step = 1
                lastf = numf

            annotations = [database[videoname]['annotations']]
            num_tubes = len(annotations)

            tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
            tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]
            for tubeid, tube in enumerate(annotations):
                # print('numf00', numf, tube['sf'], tube['ef'])
                for frame_id, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
                    label = tube['label']
                    assert actidx == label, 'Tube label and video label should be same'
                    box = tube['boxes'][frame_id, :]  # get the box as an array
                    box = box.astype(np.float32)
                    # box[2] += box[0]  #convert width to xmax
                    # box[3] += box[1]  #converst height to ymax
                    tube_labels[frame_num, tubeid] = 1 #label+1  # change label in tube_labels matrix to 1 form 0
                    tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

            possible_frame_nums = np.arange(0, lastf, step)
            # print('numf',numf,possible_frame_nums[-1])
            for frame_num in possible_frame_nums: # loop from start to last possible frame which can make a legit sequence
                frame_num = int(frame_num)
                check_tubes = tube_labels[frame_num, :]

                if np.sum(check_tubes)>0:  # check if there aren't any semi overlapping tubes
                    all_boxes = []
                    labels = []
                    image_name = imagesDir + videoname+'/{:05d}.png'.format(frame_num+1)
                    #label_name = rootpath + 'labels/' + videoname + '/{:05d}.txt'.format(frame_num + 1)
                    assert os.path.isfile(image_name), 'Image does not exist'+image_name
                    for tubeid, tube in enumerate(annotations):
                        label = tube['label']
                        if tube_labels[frame_num, tubeid]>0:
                            box = np.asarray(tube_boxes[frame_num][tubeid])
                            all_boxes.append(box)
                            labels.append(label)

                    if istrain: # if it is training video
                        trainlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                        train_action_counts[actidx] += 1 #len(labels)
                    else: # if test video and has micro-tubes with GT
                        testlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                        test_action_counts[actidx] += 1 #len(labels)
                # elif fulltest and not istrain: # if test video with no ground truth and fulltest is trues
                #     testlist.append([vid, frame_num+1, np.asarray([9999]), np.zeros((1,4))])

    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        print('train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx], int(actidx), CLASSES_JHMDB[actidx]))

    newratios = train_action_counts/5000
    #print('new   ratios', newratios)
    line = '['
    for r in newratios:
        line +='{:0.2f}, '.format(r)
    print(line+']')
    print('Trainlistlen', len(trainlist), ' testlist ', len(testlist))

    return trainlist, testlist, video_list


class JHMDBDetection(data.Dataset):
    """UCF24 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='jhmdb', input_type='rgb', full_test=False):

        self.input_type = input_type
        input_type = input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES_JHMDB
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()

        trainlist, testlist, video_list = make_lists(root, input_type, split=2, fulltest=full_test)
        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        img_name = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num)
        if os.path.exists(self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num - 1)):
            img_name_0 = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num - 1)
        else:
            img_name_0 = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num)
        if os.path.exists(self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num + 1)):
            img_name_2 = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num + 1)
        else:
            img_name_2 = self._imgpath + '/{:s}/{:05d}.png'.format(videoname, frame_num)
            # print(img_name)
        img = cv2.imread(img_name)
        height, width, channels = img.shape
        img_0 = cv2.imread(img_name_0)
        img_2 = cv2.imread(img_name_2)

        target = self.target_transform(annot_info[3], annot_info[2], width, height)

        if self.transform is not None:
            target = np.array(target)
            if self.image_set == "train":
                img, boxes, labels = self.transform(np.array([img_0, img, img_2]), target[:, :4], target[:, 4])
                image = img[1]
            else:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                img_0, _, __ = self.transform(img_0, target[:, :4], target[:, 4])
                img_2, _, __ = self.transform(img_2, target[:, :4], target[:, 4])
                img = np.array([img_0, img, img_2])
                image = img[1]
            img_0 = img[0]
            img_2 = img[2]
            _img = img[1]
            # for i in boxes:
            #     xmin = int(i[0]*300)
            #     ymin = int(i[1]*300)
            #     xmax = int(i[2]*300)
            #     ymax = int(i[3]*300)
            #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            # cv2.imwrite("0.png", image)

            image = image[:, :, (2, 1, 0)]
            _img = _img[:, :, ::-1]
            img_0 = img_0[:, :, ::-1]
            img_2 = img_2[:, :, ::-1]
            # cv2.imwrite("1.png", _img)
            # cv2.imwrite("2.png", img_0)
            # cv2.imwrite("3.png", img_2)

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(height, width,target)
        x = torch.from_numpy(np.array([img_0/255., _img/255., img_2/255., image/255.], dtype=np.float32)).permute(0, 3, 1, 2)
        return x, target, index
        # return torch.from_numpy(img), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
    return torch.stack(imgs, 0), targets, image_ids


if __name__ == '__main__':
    from utils.augmentations_ import SSDAugmentation
    from data import BaseTransform
    train_dataset = JHMDBDetection('/mnt/data/Action/data/JHMDB/', 'train', SSDAugmentation(300, (104, 117, 123)),
                                   AnnotationTransform(), input_type='rgb')
    val_dataset = JHMDBDetection('/mnt/data/Action/data/JHMDB/', 'test', BaseTransform(300, (104, 117, 123)),
                                 AnnotationTransform(), input_type='rgb',
                                 full_test=False)
    train_data_loader = data.DataLoader(train_dataset, 1, num_workers=1,
                                        shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_data_loader = data.DataLoader(val_dataset, 1, num_workers=1,
                                      shuffle=False, collate_fn=detection_collate, pin_memory=True, drop_last=True)
    for i, (images, targets, img_indexs) in enumerate(train_data_loader):
        pass