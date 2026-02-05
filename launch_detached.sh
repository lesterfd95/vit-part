docker run -d --rm --runtime=nvidia --gpus all --shm-size=32gb -it --mount type=bind,src="$(pwd)"/scripts,target=/app vit_ti:1.0 bash -c "python3 train.py --model base 2>&1 | tee -a full_train.log"
