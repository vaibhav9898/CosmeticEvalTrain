python inference.py \
    --gpu_id 0\
    --val_file_path "TSV_Files/test.tsv"\
    --final_model_path "model_checkpoints/current/resnet_Accuracy.pth"\
    --model_type "resnet" \
    --batch_size 8
