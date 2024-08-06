import csv
import numpy as np
import _pickle as cPickle
from datetime import datetime

porto_lon_range = [-9.0, -7.9]
porto_lat_range = [40.7, 41.8]

csvFile = open('./data/train.csv', 'r')
# csvFile = open('porto_sample.csv', 'r')
reader = csv.reader(csvFile)
traj_missing = []
trajectories = []
min_lon, max_lon, min_lat, max_lat = -7.0, -10.0, 43.0, 40.0
time_tra=[]
for item in reader:
    if(reader.line_num == 1):
        continue
    if(item[7] == 'True'):
        traj_missing.append(item[8])
        aaaaa=item[8]
    if(item[7] == 'False'):
        aaaaaaa=item[8][2:-2].split('],[')
        aaaa=item[0]
        trajectories.append((item[8][2:-2].split('],['),item[5],item[0]))
        # trajectories.append(item[8][2:-2].split('],['))
traj_porto = []
time_result=[]
for trajs in trajectories:
    if(len(trajs[0]) > 2):
        #print(trajs)
        Traj = []
        time1=[]
        inrange = True
        tmp_min_lon = min_lon
        tmp_max_lon = max_lon
        tmp_min_lat = min_lat
        tmp_max_lat = max_lat
        i=0
        for traj in trajs[0]:
            tr = traj.split(',')
            if (tr[0] != '' and tr[1] != ''):
                lon = float(tr[0])
                lat = float(tr[1])
                if((lat < porto_lat_range[0]) | (lat > porto_lat_range[1]) | (lon < porto_lon_range[0]) | (lon > porto_lon_range[1])):
                    inrange = False
                if(lon < tmp_min_lon):
                    tmp_min_lon = lon
                if(lon > tmp_max_lon):
                    tmp_max_lon = lon
                if(lat < tmp_min_lat):
                    tmp_min_lat = lat
                if(lat > tmp_max_lat):
                    tmp_max_lat = lat
                traj_tup = (lon, lat)
                point_time=int(trajs[-2])+i*15
                start_dtime = datetime.fromtimestamp(point_time)
                start_second = (start_dtime.hour * 3600 +
                                start_dtime.minute * 60 + start_dtime.second)
                i=i+1
                Traj.append((lon,lat,start_second,trajs[-1]))
        time1.append(trajs[-1])
    if(inrange != False):
        traj_porto.append(Traj)
        time_result.append(trajs[-1])
        min_lon = tmp_min_lon
        max_lon = tmp_max_lon
        min_lat = tmp_min_lat
        max_lat = tmp_max_lat

print(traj_porto[0])
print(len(traj_porto))
# print(min_lon)
# print(max_lon)
# print(min_lat)
# print(max_lat)

traj_w = traj_porto[:10000]
# traj_w = traj_porto[:1100]
cPickle.dump(traj_w, open('porto_trajs', 'wb'))