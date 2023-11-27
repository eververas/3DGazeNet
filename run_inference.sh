CUDA_VISIBLE_DEVICES=0 python inference.py \
                --cfg configs/inference/inference.yaml \
                --inference_data_file '/home/evangelosv/git_code/dense3Deyes/output/preprocessing/data_face68.pkl' \
                --inference_dataset_dir '/storage/nfs2/evangelosv/databases/Face/AFLW/images/' \
                --checkpoint output/models/singleview/vertex/ALL/test_0/checkpoint.pth \
                --skip_optimizer