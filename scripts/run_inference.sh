CUDA_VISIBLE_DEVICES=0 python inference.py \
                --cfg configs/inference/inference.yaml \
                --inference_data_file 'output/preprocessing/data_face68.pkl' \
                --inference_dataset_dir 'data/example_images/' \
                --checkpoint output/models/singleview/vertex/ALL/test_0/checkpoint.pth \
                --skip_optimizer