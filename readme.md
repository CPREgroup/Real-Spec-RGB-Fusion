### train a single image

`python train.py --model_dir=test/`

### use pretrained model to train a single image (inference)

*pretrained model was trained on image fake_and_real_peppers_ms from CAVE dataset. And It is not trained to converge.

`python train.py --model_dir=test/ --start_epoch=499 --end_epoch=500`

for better performance, you can set `--end_epoch=` to a larger number, like `1000`
