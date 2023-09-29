import argparse
import random
import pickle
import json
import os 




def process_dataset(window=10, mode='mixed',data_percent=1.0):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        # 'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        # 'meld':   ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        # 'EmoryNLP':['Joyful', 'Mad', 'Peaceful','Neutral', 'Sad', 'Powerful', 'Scared']
        'iemocap':['joyful', 'sad', 'neutral', 'mad', 'excited', 'fear'],
        'meld':   ['neutral', 'powerful', 'fear', 'sad', 'joyful', 'disgust', 'mad'],
        'EmoryNLP':['joyful', 'mad', 'peaceful','neutral', 'sad', 'powerful', 'fear']
    }
    label_text_set = {
        # 'iemocap':'happy, sad, neutral, angry, excited, frustrated',
        # 'meld'   :'neutral, surprise, fear, sad, joyful, disgust, angry',
        # 'EmoryNLP': 'Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared',
        # below are original emotional labelset
        'iemocap':'joyful, sad, neutral, mad, excited, fear',
        'meld'   :'neutral, powerful, fear, sad, joyful, disgust, mad',
        'EmoryNLP': 'joyful, mad, peaceful, neutral, sad, powerful, fear',
        'coarse-mixed':'joyful, sad, neutral, mad, excited, fear, disgust, peaceful, powerful'
    }
    content_concate_label = label_text_set['coarse-mixed']
    speaker_label_text_set = {
        'iemocap':'Speaker_0, Speaker_1',
        'meld':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
        'EmoryNLP':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
        'coarse-mixed': '''Speaker_0, Speaker_1, 
                       Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8, Speaker_9, 
                       Speaker_10, Speaker_11, Speaker_12, Speaker_13, Speaker_14, Speaker_15, Speaker_16, Speaker_17,'''
    }
    dataset_list = ['iemocap','meld','EmoryNLP']


    if args.mode == 'mixed':
        speaker_label_dict = {}
        content_target_dict = {}
        content_task_dict = {}
        sentence_dict = {}
        new_train_id, new_test_id, new_valid_id = [], [], []

        for dataset in dataset_list:
            new_sub_train_id = []
            data = pickle.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/text_data/{dataset}/{dataset}.pkl','rb'))

            # 不同的数据集有不同的speaker_label的处理方式
            if dataset == 'iemocap':
                all_conv_id = data[3] + data[4] + data[5]
                sentence_dict = data[2]
                for conv_id in all_conv_id:
                    temp_speaker_list = []
                    for speaker_label in data[0][conv_id]:
                        if speaker_label == 'M':
                            temp_speaker_list.append(0)
                        else:
                            temp_speaker_list.append(1)
                    speaker_label_dict[conv_id] = temp_speaker_list

            elif dataset == 'meld':
                all_conv_id = data[4] + data[5] + data[6]
                sentence_dict = data[3]
                for conv_id in all_conv_id:
                    temp_speaker_list = []
                    for speaker_label in data[0][conv_id]:
                        temp_speaker_list.append(speaker_label.index(1)+2)
                    speaker_label_dict[conv_id] = temp_speaker_list
            
            elif dataset == 'EmoryNLP':
                all_conv_id = data[3] + data[4] + data[5]
                sentence_dict = data[2]
                for conv_id in all_conv_id:
                    temp_speaker_list = []
                    for speaker_label in data[0][conv_id]:
                        temp_speaker_list.append(speaker_label.index(1)+10)
                    speaker_label_dict[conv_id] = temp_speaker_list

            # 对conversation的utterance进行处理，其中index_w用于处理窗口大小设置下面的起始index
            for conv_id in all_conv_id:
                for conv_turn in range(len(sentence_dict[conv_id])):
                    temp_content_str = 'Now you are expert of sentiment and emotional analysis. '
                    # if demons == 'True':
                    #     temp_content_str += demonstration_short[dataset]
                    temp_content_str += 'The following conversation noted between \'### ###\' involves several speakers. ### '

                    index_w = max(conv_turn-window, 0)
                    for speaker_label, sub_sent in zip(speaker_label_dict[conv_id][index_w:conv_turn+1], sentence_dict[conv_id][index_w:conv_turn+1]):
                        temp_content_str += (f'\t Speaker_{speaker_label}:"{sub_sent}"')
                        # temp_id_task_str += (f'\t Speaker_?:"{sub_sent}"')

                    content_target_dict[f'{conv_id}_{conv_turn}'] = label_set[dataset][data[1][conv_id][conv_turn]]
                    target_utterance = temp_content_str.split('\t')[-1]
                    temp_content_str += ' ### '
                    temp_content_str += f'Please select the emotional label of <{target_utterance}> from <{content_concate_label}>:'

                    # if predictions == 'True' and conv_turn > 0:
                    #     temp_predict_str = '\t'.join(temp_content_str.split('\t')[:-1])
                    #     temp_predict_str += '###'
                    #     temp_predict_str += f'Based on the above historical utterances, next utterance is spoken by <{speaker_label_dict[conv_id][conv_turn]}>, please predict the emotion states of <{speaker_label_dict[conv_id][conv_turn]}> from <{label_text_set[dataset]}>:'
                    #     temp_content_str += ('***'+temp_predict_str)
                    # elif predictions == 'True' and conv_turn == 0:
                    #     temp_content_str += ('***'+temp_content_str)
                    content_task_dict[f'{conv_id}_{conv_turn}'] = temp_content_str

            if dataset == 'iemocap':
                train_ids, test_ids, valid_ids = data[3], data[4], data[5]
            elif dataset == 'meld':
                train_ids, test_ids, valid_ids = data[4], data[5], data[6]
            elif dataset == 'EmoryNLP':
                train_ids, test_ids, valid_ids = data[3], data[4], data[5]


            # new_train_target, new_test_target, new_valid_target = [], [], []
            for train_id in train_ids:
                for conv_turn in range(len(sentence_dict[train_id])):
                    new_sub_train_id.append(f'{train_id}_{conv_turn}')
                    # if speaker_task == 'True' or speaker_task == 'True_mixed':
                    #     new_train_id.append(f'Speaker_{train_id}_{conv_turn}')
                    
                
            for test_id in test_ids:
                for conv_turn in range(len(sentence_dict[test_id])):
                    new_test_id.append(f'{test_id}_{conv_turn}')
                    # if speaker_task == 'True' or speaker_task == 'True_mixed':
                    #     new_test_id.append(f'Speaker_{test_id}_{conv_turn}')
            
            for valid_id in valid_ids:
                for conv_turn in range(len(sentence_dict[valid_id])):
                    new_valid_id.append(f'{valid_id}_{conv_turn}')
                    # if speaker_task == 'True' or speaker_task == 'True_mixed':
                    #     new_valid_id.append(f'Speaker_{valid_id}_{conv_turn}')
            num_sample = int(len(new_sub_train_id)*data_percent) #计算需要选取的样本数量
            sample = random.sample(new_sub_train_id, num_sample)          # 随机选取样本
            new_train_id.extend(sample)


                    # new_train_id =  speaker_task_dict.keys() + content_dict.keys()
            # dataset_list = ['train', 'test', 'valid']
        
        data_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/data/unified_label/{mode}'
        os.makedirs(data_path, exist_ok=True)

        with open(f'{data_path}/train.json', 'w') as f_train:
            for train_id in new_train_id:
                f_train.write(json.dumps({'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

        with open(f'{data_path}/test.json', 'w') as f_test:
            for test_id in new_test_id:
                f_test.write(json.dumps({'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

        with open(f'{data_path}/valid.json', 'w') as f_valid:
            for valid_id in new_valid_id:
                f_valid.write(json.dumps({'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')
        return data_path

# def window_process_data():
#     pass

# def perform_speaker_task():
#     pass

# def perform_domain_base_task():
    # pass

# def perform_emotion_prediction_task():
#     pass


parser = argparse.ArgumentParser(description='Data processing script')
# parser.add_argument('--dataset', type=str, default='iemocap', help='Dataset name or path')
parser.add_argument('--mode', type=str, default='mixed',choices=['mixed','sequence'], help='Dataset name or path')
parser.add_argument('--data_percent', type=float, default=1.0, help='the ratio of sampling from original dataset')
parser.add_argument('--historical_window', type=int, default=20, help='Historical window size')
parser.add_argument('--speaker_task', type=str, default='add speaker_task to main task', help='Speaker task type')
parser.add_argument('--domain_base', type=str, default='select demonstration from domain_base to add to input', help='domain_base mode')
parser.add_argument('--emotion_prediction', type=str, default='add emotion_prediction to main task', help='Emotion prediction task type')
args = parser.parse_args()



# Read dataset

# Process data
# processed_data_path = process_dataset(dataset='EmoryNLP')
# processed_data_path = process_dataset(dataset=args.dataset)
processed_data_path = process_dataset(window=args.historical_window, mode=args.mode, data_percent=args.data_percent)
# processed_data_path = process_dataset(dataset=args.dataset, window=args.historical_window, speaker_task=args.speaker_task)
print(processed_data_path)
#perform domain_base grounding
# domain_base_results = perform_domain_base_task(process_data, domain_base_task_type)

# Perform emotion prediction task
# emotion_prediction_task_results = perform_emotion_prediction_task(processed_data, emotion_prediction_task_type)

# Perform speaker task
# speaker_task_results = perform_speaker_task(processed_data, speaker_task_type)

# Output results
# output_results(speaker_task_results, emotion_prediction_task_results)
