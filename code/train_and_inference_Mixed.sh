source YOUR CONDA ENVS
source YOUR DOCKER

# The Shellparameter that controls the mainprocess
FLAG=1
# The hyperparameter you need setting: 1.MODEL_NAME, 2.Experiments_setting, 3.dataset, 4.accumulations, 5.graphics_card

# select basemodel
# MODEL_NAME='ChatGLM'
# MODEL_NAME='ChatGLM2'
# MODEL_NAME='LLaMA'
MODEL_NAME='LLaMA2'
# MODEL_NAME='Bloom-560m'

# select the experiment's model
# Experiments_setting='test'
# Experiments_setting='zero_shot'
# Experiments_setting='few_shot'
Experiments_setting='lora'
# Experiments_setting='all_parameters'

# select the dataset
# dataset='test'
# dataset='iemocap'
# dataset='meld'
# dataset='EmoryNLP'

# select the mode
dataset='mixed'
mode='mixed'
# mode='sequence'

# select the historical window for dataset
# LLaMA 's context = 1024 is enough for almost dataset, except for iemocap.
# IEMOCAP has very long conversation sample, 
# the historical window is designed for this kind of long conversation.
historical_window=12

# set the accumulation and card when backwarding and inferring
accumulations=8
graphics_card=4
BS=$((accumulations * graphics_card))


# data_percent=1.0    # 1
# data_percent=0.75    # 3/4
# data_percent=0.5    # 1/2
# data_percent=0.25   # 1/4
data_percent=0.125  # 1/8
# data_percent=0.0625 # 1/16 
# data_percent=0.01 # 1/16 
# data_percent=0.03125 # 1/32 
# data_percent=0.015625 # 1/64 
echo "data_percent: ${data_percent}"


case ${MODEL_NAME} in
'ChatGLM'|'ChatGLM2'|'LLaMA'|'LLaMA2'|'Bloom-560m')
    case ${Experiments_setting} in
    'zero_shot'|'few_shot'|'lora'|'all_parameters')
        # case ${dataset} in
        # 'iemocap'|'meld'|'EmoryNLP')
        case ${mode} in
        'mixed'|'sequence')
            echo "******************************************************************************************"
            echo "All parameters are valid."
            echo "The mode you have selected is: ${mode} !"
            echo "The base model you have selected is ${MODEL_NAME}!"
            echo "The model's SFT method you have selected: ${Experiments_setting}!"
            echo "******************************************************************************************"
            ;;
        *)
            echo "The dataset parameter is invalid. CHECK IT OUT!"
            FLAG=0
            ;;
        esac
        ;;
    *)
        echo "The Experiments_setting parameter is invalid. CHECK IT OUT!"
        FLAG=0
        ;;
    esac
    ;;
*)
    echo "The MODEL_NAME parameter is invalid. CHECK IT OUT!"
    FLAG=0
    ;;
esac

# MAX_LENGTH=1200 
if [ ${FLAG} = 1 ]
then
    DATA_PATH=$(python data_process_mixed.py --mode ${mode} \
    --historical_window ${historical_window} \
    --data_percent ${data_percent})
    # DATA_PATH=$(python data_process.py --dataset ${dataset} \
    #     --historical_window ${historical_window} \
    #     --speaker_task ${speaker_task} \
    #     --domain_base ${domain_base} \
    #     --emotion_prediction ${emotion_prediction})
    # if [ $? -eq 0 ]; then
    #     echo "******************************************************************************************"
    #     echo -e "Data procession has executed successfully !"
    #     echo "******************************************************************************************"

    # else
    #     echo "Data procession script encountered an error."
    # fi
    #     # DATA_PATH=./construct_dataset/${Experiments_setting}/${dataset}
    
    if [ ${dataset} = 'mixed' ]
    then
        MAX_LENGTH=1024
    elif [ ${dataset} = 'iemocap' ]    
    then
        MAX_LENGTH=1200
    elif [ ${dataset} = 'meld' ]
    then
        MAX_LENGTH=1024
    elif [ ${dataset} = 'EmoryNLP' ]
    then
        MAX_LENGTH=1024
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo "******************************************************************************************"
    echo -e "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
    echo "******************************************************************************************"


    if [ ${MODEL_NAME} = 'ChatGLM' ]
    then
        MODEL_PATH='./LLM_bases/chatglm-6b'
    elif [ ${MODEL_NAME} = 'ChatGLM2' ]
    then
        MODEL_PATH='./LLM_bases/chatglm2-6b'
    elif [ ${MODEL_NAME} = 'LLaMA' ]
    then
        MODEL_PATH='./LLM_bases/llama-7b-hf'
    elif [ ${MODEL_NAME} = 'LLaMA2' ]
    then
        MODEL_PATH='./LLM_bases/llama-2-7b'
    elif [ ${MODEL_NAME} = 'Bloom-560m' ]    
    then
        MODEL_PATH='./LLM_bases/bloom-560m'
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo -e "Your choose ${MODEL_NAME}! Model Parameters should be initialized in the path \n ${MODEL_PATH}"


    if [ ${Experiments_setting} = 'zero_shot' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=False
        LR=0
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as ZERO_SHOT model"
    elif [ ${Experiments_setting} = 'few_shot' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=False
        LR=0
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as FEW_SHOT model"
    elif [ ${Experiments_setting} = 'lora' ]
    then
        DO_EVAL=True
        DO_TRAIN=True
        LORA=True
        LR=2e-4
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as LORA model"
    elif [ ${Experiments_setting} = 'all_parameters' ]
    then
        DO_EVAL=True
        DO_TRAIN=True
        LORA=False
        LR=2e-5
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as ALL_PARAMETERS model"
    else
        echo -e "Your choose is not in MY candidations! Please CHECK your Experiments Setting!"
    fi


    if [ ${mode} = 'mixed' ]
    then
        echo "Processed Data_Path: $DATA_PATH"
        deepspeed --master_port=29500 main_Unilabel.py \
        --dataset ${mode} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_PATH} \
        --output_dir ./experiments/${mode}/${MODEL_NAME}/${Experiments_setting}_win${historical_window}_data_${data_percent}/ \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ./code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 8 \
        --num_train_epochs 8 \
        --save_steps 100000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_eval ${DO_EVAL} \
        --do_train ${DO_TRAIN} \
        # --data_percent ${data_percent}
        # --checkpoint_dir ${CHECKPOINT_DIR} \
        # --zero_shot ${ZERO_SHOT}

    elif [ ${speaker_task} = 'sequence' ]
    then
        # echo "${DATA_PATH}"
        DATA_SPEAKER_PATH=$(echo "$DATA_PATH" | cut -d ',' -f 1)
        DATA_WINDOW_PATH=$(echo "$DATA_PATH" | cut -d ',' -f 2)
        Speaker_Model_output_dir=./experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/${speaker_task}_one
        Content_Model_output_dir=./experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/${speaker_task}_two
        echo "*********************************************"
        echo -e "\n DATA_SPEAKER_PATH: $DATA_SPEAKER_PATH \n"
        echo "*********************************************"
        echo -e "\n DATA_WINDOW_PATH: $DATA_WINDOW_PATH \n"
        echo "*********************************************"
        echo -e "\n Speaker_Model_output_dir: $Speaker_Model_output_dir \n"
        echo "*********************************************"
        echo -e "\n Content_Model_output_dir: $Content_Model_output_dir \n"

        echo "*********************************************"
        echo "Start to train on Speaker Identification task!"
        echo "*********************************************"
        # deepspeed --master_port=29500 main_new.py \
        # --dataset ${dataset} \
        # --model_name_or_path ${MODEL_PATH} \
        # --data_dir ${DATA_SPEAKER_PATH} \
        # --output_dir ${Speaker_Model_output_dir} \
        # --max_length ${MAX_LENGTH} \
        # --batch_size ${BS} \
        # --deepspeed_config ./code/data_utils/deepspeed_config.json \
        # --gradient_accumulation_steps ${accumulations} \
        # --eval_batch_size 8 \
        # --num_train_epochs 3 \
        # --save_steps 100000 \
        # --lora ${LORA}\
        # --learning_rate ${LR} \
        # --do_train ${DO_TRAIN} \
        # --do_eval ${DO_EVAL} \
        # --statistic_mode False
        # --checkpoint_dir ${CHECKPOINT_DIR} \
        # --zero_shot ${ZERO_SHOT}
        echo "*******************************************************************"
        echo "Speaker Identification task has been achieved successfully!"
        echo "*******************************************************************"

        echo "*********************************************"
        echo "Start to train on Emotion Recognition task!"
        echo "*********************************************"
        deepspeed --master_port=29500 main_new.py \
        --dataset ${dataset} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_WINDOW_PATH} \
        --output_dir ${Content_Model_output_dir} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ./code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 16 \
        --num_train_epochs 15 \
        --save_steps 100000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_train ${DO_TRAIN} \
        --do_eval ${DO_EVAL} \
        --statistic_mode True \
        --checkpoint_dir ${Speaker_Model_output_dir} 

    fi
fi