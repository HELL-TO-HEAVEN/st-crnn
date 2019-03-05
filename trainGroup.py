from geo_data_decoder import *
from eval_tools import *
import config
from rnn_model_keras import create_group_rnn_text_model

PRETRAINED_FS = config.PRETRAINED_FS
PRETRAINED_LA = config.PRETRAINED_LA

def train(dataset = 'FS'):

    if dataset == 'FS':
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec = geo_data_clean_fs()
        print(len(user_feature_sequence.keys()))
        train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index \
            = geo_dataset_train_test_text(user_feature_sequence, useful_vec, seg_max_record)
        print ("Feature generation completed")
        nearest_location_last(vali_X, vali_evl, center_location_list)
        model = geo_lprnn_trainable_text_model(user_dim, seg_max_record, word_vec)
        model.load_weights(PRETRAINED_FS)
        all_output_array = model.predict(vali_X)
        evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
        print ("Train_x[0] shape:", train_X[1].shape)
        print ("Train_x[0] shape:", train_X[2].shape)
        print ("Train_Y shape:", train_Y.shape)
        geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y, vali_evl, model, center_location_list, word_index,
                                 dataset='FS_trainable_')

    elif dataset=='LA':
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec= geo_data_clean_la()
        print (len(user_feature_sequence.keys()))

        groups = 80
        user_group_weight=[len(user_feature_sequence.keys())][groups]
        for u in user_feature_sequence.keys():
            np.set_printoptions(precision=2)
            user_group_weight[u] = np.random(groups)
            weigth_sum = sum(user_group_weight[u])
            user_group_weight[u] = user_group_weight[u]/weigth_sum #normlization

        iters = 10
        for it in range(iters -1):
            p_vec=[]
            for model_idx in range(groups-1):
                train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index,train_user\
                    = group_geo_dataset_train_test_text(user_group_weight,model_idx,user_feature_sequence,useful_vec, seg_max_record)
                print ("Feature generation completed")
                #frequent_location_last(train_X, vali_X, vali_evl, center_location_list)
                nearest_location_last(vali_X, vali_evl, center_location_list)
                model =create_group_rnn_text_model(user_dim,seg_max_record,word_vec)
                #model.load_weights(PRETRAINED_LA)
                # all_output_array = model.predict(vali_X)
                # evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
                print ("Train_x[0] shape:", train_X[1].shape)
                print ("Train_x[0] shape:", train_X[2].shape)
                print ("Train_Y shape:", train_Y.shape)
                geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y, vali_evl, model, center_location_list,word_index,
                                         dataset='LA',epoch=5)
                all_output_array = model.predict(train_X)
                p_vec.append(evaluation_group_probablity(all_output_array, train_evl, center_location_list,train_user))
            p_vec = p_vec/sum(p_vec)
        # 迭代



if __name__ == '__main__':
    #train(dataset='FS')
    train(dataset='LA')