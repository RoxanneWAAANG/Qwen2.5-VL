import kagglehub

# Download latest version
path = kagglehub.dataset_download("deathtrooper/multichannel-glaucoma-benchmark-dataset")

print("Path to dataset files:", path)
# Path to dataset files: /home/jack/.cache/kagglehub/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/versions/10