import os

splits_file_dir = '/home/user/git/realtime-action-detection/data/jhmdb/splits'
splits = os.listdir(splits_file_dir)
final_splits = {}

def read_split(path, action_class):
    train = []
    test = []
    with open(path) as f:
        for line in f.readlines():
            line = line.split(" ")
            if line[1] == "1\n":
                train.append(os.path.join(action_class, line[0]+"\n"))
            else:
                test.append(os.path.join(action_class, line[0]+'\n'))
    return train, test

for file in splits:
    file_info = file.split(".")[0]
    file_info = file_info.split("_")
    file_split = file_info[-1]
    file_class = "_".join(file_info[:-2])
    train, test = read_split(os.path.join(splits_file_dir, file), file_class)
    if file_split not in final_splits:
        final_splits[file_split] = {'train': [], 'test': []}
    final_splits[file_split]['train'] += train
    final_splits[file_split]['test'] += test
for i in range(3):
    with open("trainlist{:02d}.txt".format(i+1), 'w') as f:
        f.writelines(final_splits['split{}'.format(i+1)]['train'])
        with open("testlist{:02d}.txt".format(i+1), 'w') as f:
            f.writelines(final_splits['split{}'.format(i+1)]['test'])