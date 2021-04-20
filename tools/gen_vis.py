import os

# make folder
folder = './vis_best_with_ori_camid_track_rank'
if not os.path.exists(folder):
    os.makedirs(folder, mode = 0o777)

# read results
file_path = '/home/cybercore/su/AICity2021-VOC-ReID/output/aicity20/0409-ensemble/A-Nam-SynData-next101-320-circle/track2.txt'
file_submit = open(file_path, 'r')
all_id_list = file_submit.readlines()


for idx, id_list in enumerate(all_id_list):
    
    id_list = id_list.split(' ')
    file = open(f'{folder}/{idx+1:06}.txt', 'w')
    for id in id_list:
        id = id.split('.')[0]
        file.write(id + '\n')
    
    file.close()

