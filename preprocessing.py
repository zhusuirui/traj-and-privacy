from tools import preprocess
from tools.distance_compution import trajectory_distance_combain,trajecotry_distance_list,\
    trajecotry_distance_list_time,trajectory_distance_combain_time
import _pickle as cPickle
import numpy as  np


def distance_comp(coor_path):
    # traj_coord = cPickle.load(open(coor_path, 'r'))[0]
    with open(coor_path, 'rb') as f:
        traj_coord = cPickle.load(f, encoding='bytes')
    traj_coord =traj_coord[0]
    np_traj_coord = []
    np_traj_time=[]
    for t in traj_coord:
        temp_coord=[]
        temp_time=[]
        for item in t:
            temp_coord.append([item[0],item[1]])
            temp_time.append([float(item[2]),float(0)])
        np_traj_coord.append(np.array(temp_coord))
        np_traj_time.append(np.array(temp_time))
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    distance_type = 'hausdorff'

    trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type,
                             data_name=data_name)
    trajecotry_distance_list_time(np_traj_time, batch_size=200, processors=15, distance_type=distance_type,
                             data_name=data_name)
    trajectory_distance_combain(3000, batch_size=200, metric_type=distance_type, data_name=data_name)
    trajectory_distance_combain_time(3000, batch_size=200, metric_type=distance_type, data_name=data_name)

    path1="features/porto_hausdorff_distance_all_3000"
    spatial1 = cPickle.load(open(path1, 'rb'))
    max_dis_s2=spatial1.max()

    path2="features/porto_hausdorff_distance_all_time3000"
    time1 = cPickle.load(open(path2, 'rb'))
    max_dis_t2=time1.max()

    spatial2 = spatial1 / max_dis_s2
    spatial3 = np.exp(-spatial2 * 8)
    time2 = time1 / max_dis_t2
    time3 = np.exp(-time2 * 8)

    result_time_spatial=0.5*np.array(time3)+0.5*np.array(spatial3)
    cPickle.dump(result_time_spatial,open('./features/'+
            data_name+'_'+'hausdorff_'+'distance_allin_'+'3000', 'wb'))

if __name__ == '__main__':
    coor_path, data_name = preprocess.trajectory_feature_generation(path= './porto_trajs')
    distance_comp(coor_path)
