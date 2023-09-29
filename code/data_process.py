import argparse
import pickle
import json
import os 




def process_dataset(dataset, window=110, speaker_task='True', demons='False', predictions='True'):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld':   ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'EmoryNLP':['Joyful', 'Mad', 'Peaceful','Neutral', 'Sad', 'Powerful', 'Scared']
    }
    label_text_set = {
        'iemocap':'happy, sad, neutral, angry, excited, frustrated',
        'meld'   :'neutral, surprise, fear, sad, joyful, disgust, angry',
        'EmoryNLP':'Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared'
    }
    speaker_label_text_set = {
        'iemocap':'Speaker_0, Speaker_1',
        'meld':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
        'EmoryNLP':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
    }



    emotional_dict = {text_label:num_label for num_label, text_label in enumerate(label_set[dataset])}
    speaker_label_dict = {}
    content_target_dict = {}
    speaker_target_dict = {}
    content_task_dict = {}
    speaker_task_dict = {}
    sentence_dict = {}
    data = pickle.load(open(f'YOUR_DATASET_COLLECTIONS_FOR_ERC_PATH/{dataset}/{dataset}.pkl','rb'))

    # 不同的数据集有不同的speaker_label的处理方式
    #Different datasets have different ways of handling speaker_label
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
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list
    
    elif dataset == 'EmoryNLP':
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list

    # 对conversation的utterance进行处理，其中index_w用于处理窗口大小设置下面的起始index
    # Process the utterances in the conversation, where 'index_w' is used to handle the starting index under the window size setting.
    for conv_id in all_conv_id:
        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = 'Now you are expert of sentiment and emotional analysis. '
            if demons == 'True':
                temp_content_str += demonstration_short[dataset]
            temp_content_str += 'The following conversation noted between \'### ###\' involves several speakers. ### '

            index_w = max(conv_turn-window, 0)
            for speaker_label, sub_sent in zip(speaker_label_dict[conv_id][index_w:conv_turn+1], sentence_dict[conv_id][index_w:conv_turn+1]):
                temp_content_str += (f'\t Speaker_{speaker_label}:"{sub_sent}"')
                # temp_id_task_str += (f'\t Speaker_?:"{sub_sent}"')

            content_target_dict[f'{conv_id}_{conv_turn}'] = label_set[dataset][data[1][conv_id][conv_turn]]
            target_utterance = temp_content_str.split('\t')[-1]
            temp_content_str += ' ### '
            temp_content_str += f'Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}>:'
            # demon
            # temp_content_str += f'Here is a demonstration. {demon}'
            if predictions == 'True' and conv_turn > 0:
                temp_predict_str = '\t'.join(temp_content_str.split('\t')[:-1])
                temp_predict_str += '###'
                temp_predict_str += f'Based on the above historical utterances, next utterance is spoken by <{speaker_label_dict[conv_id][conv_turn]}>, please predict the emotion states of <{speaker_label_dict[conv_id][conv_turn]}> from <{label_text_set[dataset]}>:'
                temp_content_str += ('***'+temp_predict_str)
            elif predictions == 'True' and conv_turn == 0:
                temp_content_str += ('***'+temp_content_str)
            content_task_dict[f'{conv_id}_{conv_turn}'] = temp_content_str

    for conv_id in all_conv_id:
        for conv_turn in range(len(data[2][conv_id])):
            temp_speaker_task_str = f'Now you are expert of sentiment and emotional analysis. Please select the Speaker label of the utterance <Speaker: {sentence_dict[conv_id][conv_turn]}> from <{speaker_label_text_set[dataset]}>:'
            speaker_target_dict[f'Speaker_{conv_id}_{conv_turn}'] = f'Speaker_{speaker_label_dict[conv_id][conv_turn]}'
            speaker_task_dict[f'Speaker_{conv_id}_{conv_turn}'] = temp_speaker_task_str

    if dataset == 'iemocap':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]
    elif dataset == 'meld':
        train_ids, test_ids, valid_ids = data[4], data[5], data[6]
    elif dataset == 'EmoryNLP':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]


    new_train_id, new_test_id, new_valid_id = [], [], []
    # new_train_target, new_test_target, new_valid_target = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f'{train_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_train_id.append(f'Speaker_{train_id}_{conv_turn}')
            
        
    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f'{test_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_test_id.append(f'Speaker_{test_id}_{conv_turn}')
    
    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f'{valid_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_valid_id.append(f'Speaker_{valid_id}_{conv_turn}')


            # new_train_id =  speaker_task_dict.keys() + content_dict.keys()
    # dataset_list = ['train', 'test', 'valid']
    if predictions == 'False':
        if speaker_task == 'True_mixed':
            data_path = f'YOUR_PROCESSED_DATASET_COLLECTIONS_FOR_ERC_PATH/{dataset}/speaker_window'
            os.makedirs(data_path, exist_ok=True)
            with open(f'{data_path}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps({'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')
                    else:
                        f_train.write(json.dumps({'input':f'{speaker_task_dict[train_id]}','target':f'{speaker_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

            with open(f'{data_path}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps({'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')
                    # else:
                        # f_test.write(json.dumps({'input':f'{speaker_task_dict[test_id]}','target':f'{speaker_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

            with open(f'{data_path}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps({'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')
                    else:
                        f_valid.write(json.dumps({'input':f'{speaker_task_dict[valid_id]}','target':f'{speaker_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')

        elif speaker_task == 'True':
            data_path_speaker = f'YOUR_PROCESSED_DATASET_COLLECTIONS_FOR_ERC_PATH{dataset}/speaker'
            os.makedirs(data_path_speaker, exist_ok=True)
            with open(f'{data_path_speaker}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in speaker_task_dict:
                        f_train.write(json.dumps({'input':f'{speaker_task_dict[train_id]}','target':f'{speaker_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

            with open(f'{data_path_speaker}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in speaker_task_dict:
                        f_test.write(json.dumps({'input':f'{speaker_task_dict[test_id]}','target':f'{speaker_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')
            
            with open(f'{data_path_speaker}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in speaker_task_dict:
                        f_valid.write(json.dumps({'input':f'{speaker_task_dict[valid_id]}','target':f'{speaker_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')

            data_path_window = f'YOUR_PROCESSED_DATASET_COLLECTIONS_FOR_ERC_PATH/{dataset}/window'
            os.makedirs(data_path_window, exist_ok=True)
            with open(f'{data_path_window}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps({'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

            with open(f'{data_path_window}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps({'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')


            with open(f'{data_path_window}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps({'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')

            data_path = data_path_speaker + "," +data_path_window         
                        

        elif speaker_task == 'None' and demons == 'False':
            data_path = f'YOUR_PROCESSED_DATASET_COLLECTIONS_FOR_ERC_PATH/{dataset}/window'
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
        
        elif speaker_task == 'None' and demons == 'True':
            data_path = f'YOUR_PROCESSED_DATASET_COLLECTIONS_FOR_ERC_PATH/{dataset}/demon'
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
    elif predictions == 'True':
        if speaker_task == 'None' and demons == 'False':
            data_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/data/{dataset}/predict/window'
        elif speaker_task == 'None' and demons == 'True':
            data_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/data/{dataset}/predict/demon'
        os.makedirs(data_path, exist_ok=True)
            
        with open(f'{data_path}/train.json', 'w') as f_train:
            for train_id in new_train_id:
                f_train.write(json.dumps({'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

        with open(f'{data_path}/test.json', 'w') as f_test:
            for test_id in new_test_id:
                content_task_dict[test_id] = content_task_dict[test_id].split('***')[0]
                f_test.write(json.dumps({'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

        with open(f'{data_path}/valid.json', 'w') as f_valid:
            for valid_id in new_valid_id:
                content_task_dict[valid_id] = content_task_dict[valid_id].split('***')[0]
                f_valid.write(json.dumps({'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')
    return data_path

# def window_process_data():
#     pass

# def perform_speaker_task():
#     pass

# def perform_domain_base_task():
    # pass

def perform_emotion_prediction_task():
    pass


parser = argparse.ArgumentParser(description='Data processing script')
parser.add_argument('--dataset', type=str, default='iemocap', help='Dataset name or path')
parser.add_argument('--historical_window', type=int, default=20, help='Historical window size')
parser.add_argument('--speaker_task', type=str, default='add speaker_task to main task', help='Speaker task type')
parser.add_argument('--domain_base', type=str, default='select demonstration from domain_base to add to input', help='domain_base mode')
parser.add_argument('--emotion_prediction', type=str, default='add emotion_prediction to main task', help='Emotion prediction task type')
args = parser.parse_args()




# Process data
processed_data_path = process_dataset(dataset=args.dataset, window=args.historical_window, speaker_task=args.speaker_task, demons=args.domain_base, predictions=args.emotion_prediction)

print(processed_data_path)

