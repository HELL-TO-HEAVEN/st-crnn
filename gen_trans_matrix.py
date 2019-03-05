import config
import io
import numpy as np
from geo_data_decoder import *
from eval_tools import *

TWEET_PATH = config.TWEET_PATH
POI_PATH = config.POI_PATH
LA_TWEETS = config.LA_TWEETS
GRID_COUNT = config.GRID_COUNT

def process_data_la():
    checkin_inter = io.open("d:\\checkin_inter", encoding='utf-8', mode='w')
    # poi_index_dict:<checkin_id,area_idx>
    user_record_sequence, poi_index_dict, center_location_list= geo_data_clean_la_without_featrue()
    print(GRID_COUNT)

    global_local_trans_matrix = np.zeros( (GRID_COUNT,GRID_COUNT) )
    day_split = 2
    time_spans = int(24/day_split)

    time_local_trans_matrix = np.zeros( ( time_spans,GRID_COUNT,GRID_COUNT )  )


    for i in range(GRID_COUNT):
        global_local_trans_matrix[i][i]=1
        for j in range(time_local_trans_matrix.shape[0]):
            time_local_trans_matrix[j][i][i]=1

    useful_poi_dict={}
    # generate useful pois
    for user in user_record_sequence.keys():
        trajs = user_record_sequence[user]
        for traj in trajs:
            pre_area_i=None
            pre_area_j = None
            for record in traj:
                if record[0] in poi_index_dict.keys():
                    area_idx = poi_index_dict[record[0]]
                    hour = time_hour(record[4])
                    hour_interval = int(hour/day_split)

                    # print("====inter %s,%s, %s" % (record[4],hour,hour_interval) )
                    area_idx_i=int(area_idx/GRID_COUNT)
                    area_idx_j = area_idx - area_idx_i*GRID_COUNT
                    global_local_trans_matrix[area_idx_i][area_idx_j] += 1

                    time_local_trans_matrix[hour_interval][area_idx_i][area_idx_j] += 1

                    if pre_area_i is not None:
                        checkin_inter.write(str(abs(area_idx_i-pre_area_i))+" "+str(abs(area_idx_j-pre_area_j))+"\n")

                    pre_area_i = area_idx_i
                    pre_area_j = area_idx_j
                    # if pre_area_idx is not None:
                    #     global_local_trans_matrix[pre_area_idx][area_idx]+=1
                    #     # print("====global_local_trans_matrix %s,%s, %s" % (pre_area_idx, area_idx, global_local_trans_matrix[pre_area_idx][area_idx]))
                    #     time_local_trans_matrix[hour_interval][pre_area_idx][area_idx]+=1
                    # pre_area_idx = area_idx

    checkin_inter.close()

    log_out = io.open("d:\\log", encoding='utf-8', mode='w')
    np.set_printoptions(threshold=np.inf)
    print("====global_local_trans_matrix")
    for i in range(global_local_trans_matrix.shape[0]):
        # print(global_local_trans_matrix[i])
        for j in range(global_local_trans_matrix[i].shape[0]):
            log_out.write(" "+str(global_local_trans_matrix[i][j]))
        log_out.write('\n')
    log_out.close()

    print("====time_local_trans_matrix")

    t_logs=[]
    for k in range(time_spans):
        name = "d:\\z-data\\t" + str(k) + "log"
        print(name)
        t_logs.append(io.open(name, encoding='utf-8', mode='w'))

    for k in range(len(t_logs)):
        for i in range(time_local_trans_matrix[k].shape[0]):
            # print(global_local_trans_matrix[i])
            print(t_logs[k].name)
            for j in range(time_local_trans_matrix[k][i].shape[0]):
                t_logs[k].write(" " + str(time_local_trans_matrix[k][i][j]))
            t_logs[k].write('\n')
        t_logs[k].close()
    # for i in range(time_local_trans_matrix.shape[0]):
    #     for j in range(time_local_trans_matrix[i].shape[0]):
    #         print(time_local_trans_matrix[i][j])
    print("====done===")

if __name__ == '__main__':
    process_data_la()