source YOUR CONDA ENVS
source YOUR DOCKER

# The Shellparameter that controls the mainprocess
FLAG=1
# The hyperparameter you need setting: 1.MODEL_NAME, 2.Experiments_setting, 3.dataset, 4.accumulations, 5.graphics_card

# select basemodel
MODEL_NAME='ChatGLM'
# MODEL_NAME='ChatGLM2'
# MODEL_NAME='LLaMA'
# MODEL_NAME='LLaMA2'
# MODEL_NAME='Bloom-560m'

# select the experiment's model
# Experiments_setting='test'
# Experiments_setting='zero_shot'
# Experiments_setting='few_shot'
Experiments_setting='lora'
# Experiments_setting='all_parameters'

# select the dataset
# dataset='test'
dataset='iemocap'
# dataset='meld'
# dataset='EmoryNLP'

# select the historical window for dataset
# LLaMA 's context = 1024 is enough for almost dataset, except for iemocap.
# IEMOCAP has very long conversation sample, 
# the historical window is designed for this kind of long conversation.
# historical_window=20

# set the accumulation and card when backwarding and inferring
accumulations=8
graphics_card=2
BS=$((accumulations * graphics_card))

# parameter that determines whether the speaker_identification task is add to train stage
# meanwhile the speaker_identification loss is also added to the total loss 
# (actually another same next token prediction loss)
# speaker_task has three options[True, True_mixed, None]
# speaker_task='True' 
# speaker_task='True_mixed'
# speaker_task='None'
# echo "speaker_task: ${speaker_task}"
# True means storing the processed data separately (two Stage training)
# True_mixed means storing the processed data Unifiedly (One stage training)
# None means no speaker identification task in main task (only processed in window mode)

# domain_base='True'
# domain_base='False'
# echo "domain_base: ${domain_base}"



# parameter that determines whether the emotion_prediction task is added to train stage, 
# meanwhile the KL divergence is added to the total loss
# emotion_prediction='True'
# emotion_prediction='False'
# echo "emotion_prediction: ${emotion_prediction}"

case ${MODEL_NAME} in
'ChatGLM'|'ChatGLM2'|'LLaMA'|'LLaMA2'|'Bloom-560m')
    case ${Experiments_setting} in
    'zero_shot'|'few_shot'|'lora'|'all_parameters')
        case ${dataset} in
        'iemocap'|'meld'|'EmoryNLP')
            echo "******************************************************************************************"
            echo "All parameters are valid."
            echo "The dataset you have selected is: ${dataset} !"
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
    DATA_PATH=$(python data_process_plain.py --dataset ${dataset})
    if [ $? -eq 0 ]; then
        echo "******************************************************************************************"
        echo -e "Data procession has executed successfully !"
        echo "******************************************************************************************"

    else
        echo "Data procession script encountered an error."
    fi
        # DATA_PATH=leishanglin/LLMs_for_ERC/construct_dataset/${Experiments_setting}/${dataset}

    if [ ${dataset} = 'iemocap' ]    
    then
        MAX_LENGTH=300
    elif [ ${dataset} = 'meld' ]
    then
        MAX_LENGTH=300
    elif [ ${dataset} = 'EmoryNLP' ]
    then
        MAX_LENGTH=300
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo "******************************************************************************************"
    echo -e "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
    echo "******************************************************************************************"


    if [ ${MODEL_NAME} = 'ChatGLM' ]
    then
        MODEL_PATH='./LLM_bases/chatglm-6b/'
    elif [ ${MODEL_NAME} = 'ChatGLM2' ]
    then
        MODEL_PATH='./LLM_bases/chatglm2-6b/'
    elif [ ${MODEL_NAME} = 'LLaMA' ]
    then
        MODEL_PATH='./LLM_bases/llama-7b-hf/'
    elif [ ${MODEL_NAME} = 'LLaMA2' ]
    then
        MODEL_PATH='./LLM_bases/llama-2-7b/'
    elif [ ${MODEL_NAME} = 'Bloom-560m' ]    
    then
        MODEL_PATH='./LLM_bases/bloom-560m/'
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


    echo "Lora: ${LORA}"
    echo "Processed Data_Path: $DATA_PATH"
    deepspeed --master_port=29500 main_new.py \
    --dataset ${dataset} \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ${DATA_PATH} \
    --output_dir ./experiments/${MODEL_NAME}/Plain/${Experiments_setting}/${dataset}/${speaker_task} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BS} \
    --deepspeed_config ./code/data_utils/deepspeed_config.json \
    --gradient_accumulation_steps ${accumulations} \
    --eval_batch_size 8 \
    --num_train_epochs 6 \
    --save_steps 100000 \
    --lora ${LORA}\
    --learning_rate ${LR} \
    --do_eval ${DO_EVAL} \
    --do_train ${DO_TRAIN} \
    # --data_percent ${percent} 
    # --gradient_checkpointing \
    # --checkpoint_dir ${CHECKPOINT_DIR} \
    # --zero_shot ${ZERO_SHOT}
fi