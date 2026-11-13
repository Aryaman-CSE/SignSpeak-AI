import os, urllib.request, tarfile, re

base = os.path.abspath('SignSpeak-AI-main' if os.path.exists('SignSpeak-AI-main/Tensorflow') else '.')
pre_dir = os.path.join(base, 'Tensorflow', 'workspace', 'pre-trained-models')
os.makedirs(pre_dir, exist_ok=True)

url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
tar_path = os.path.join(pre_dir, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
extracted_dir = os.path.join(pre_dir, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')

if not os.path.exists(extracted_dir):
    print('[INFO] Downloading COCO Pretrained SSD MobileNet V2 FPNLite (320x320)...')
    urllib.request.urlretrieve(url, tar_path)
    print('[INFO] Extracting pretrained model...')
    with tarfile.open(tar_path) as tar:
        tar.extractall(pre_dir)

# Update pipeline.config with exact absolute paths and num_classes = 26
cfg_path = os.path.join(base, 'Tensorflow', 'workspace', 'models', 'pipeline.config')
cfg = open(cfg_path, 'r', encoding='utf-8').read()

ckpt_path = os.path.join(extracted_dir, 'checkpoint', 'ckpt-0').replace('\\', '/')
lbl_path = os.path.join(base, 'Tensorflow', 'workspace', 'annotations', 'label_map.pbtxt').replace('\\', '/')
train_rec = os.path.join(base, 'Tensorflow', 'workspace', 'annotations', 'train.record').replace('\\', '/')
test_rec = os.path.join(base, 'Tensorflow', 'workspace', 'annotations', 'test.record').replace('\\', '/')

cfg = re.sub(r'num_classes:\s*\d+', 'num_classes: 26', cfg)
cfg = re.sub(r'batch_size:\s*\d+', 'batch_size: 8', cfg)
cfg = re.sub(r'fine_tune_checkpoint:\s*"[^"]*"', f'fine_tune_checkpoint: "{ckpt_path}"', cfg)
cfg = re.sub(r'fine_tune_checkpoint_type:\s*"[^"]*"', 'fine_tune_checkpoint_type: "detection"', cfg)
cfg = re.sub(r'label_map_path:\s*"[^"]*"', f'label_map_path: "{lbl_path}"', cfg)
cfg = re.sub(r'(train_input_reader[\s\S]*?input_path:\s*)"[^"]*"', rf'\1"{train_rec}"', cfg, count=1)
cfg = re.sub(r'(eval_input_reader[\s\S]*?input_path:\s*)"[^"]*"', rf'\1"{test_rec}"', cfg, count=1)

with open(cfg_path, 'w', encoding='utf-8') as f:
    f.write(cfg)

print('[SUCCESS] Pretrained model ready & pipeline.config updated for 26 classes!')
