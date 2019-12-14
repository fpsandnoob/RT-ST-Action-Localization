import os
# import tqdm


def getData(dir_path, mode='train'):
    def read_csv(path):
        with open(path, 'r') as f:
            return f.readlines()

    def getOrderDict(data: list, data_excluded: list):
        orderdict = []
        line = None
        for l in data:  # type: str
            d = l.rstrip().split(",")
            video_id = d[0]
            frame_id = d[1]
            if ",".join([video_id,frame_id]) in data_excluded:
                continue
            path = "%s" % video_id
            frame_id = "%d" % int(frame_id)
            box_x1 = d[2]
            box_y1 = d[3]
            box_x2 = d[4]
            box_y2 = d[5]
            class_index = int(d[-1]) - 1
            assert class_index >= 0
            if line is None:
                line = [path, frame_id, ",".join([box_x1, box_y1, box_x2, box_y2, str(class_index)])]
            elif path == line[0]:
                line.append(",".join([box_x1, box_y1, box_x2, box_y2, str(class_index)]))
            elif path != line[0]:
                orderdict.append(" ".join(line) + "\n")
                line = [path, frame_id, ",".join([box_x1, box_y1, box_x2, box_y2, str(class_index)])]
        return orderdict

    excluded_path = os.path.join(dir_path, "ava_{}_excluded_timestamps_v2.1.csv".format(mode))
    path = os.path.join(dir_path, "ava_{}_v2.1.csv".format(mode))
    data = read_csv(path)
    data_excluded = read_csv(excluded_path)
    dict = getOrderDict(data, data_excluded)
    with open("anno_{}.txt".format(mode), 'w') as f:
        f.writelines(dict)


getData("./")
getData("./", mode='val')
