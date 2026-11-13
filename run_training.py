import os, subprocess, sys
import tensorflow as tf
tf.get_logger().setLevel('INFO')

base = 'SignSpeak-AI-main' if os.path.exists('SignSpeak-AI-main/Tensorflow') else '.'
script = os.path.join(base, 'scripts', 'model_main_tf2.py')
model_dir = os.path.join(base, 'Tensorflow', 'workspace', 'models', 'my_ssd_mobnet')
cfg = os.path.join(base, 'Tensorflow', 'workspace', 'models', 'pipeline.config')
os.makedirs(model_dir, exist_ok=True)

cmd = [
    sys.executable, script,
    f'--model_dir={model_dir}',
    f'--pipeline_config_path={cfg}',
    '--num_train_steps=2000',
    '--checkpoint_every_n=200'
]
print('[INFO] Starting SignSpeak AI Fine-Tuning (Saves checkpoint every 200 steps)...')
print('[INFO] Wait ~60 seconds for the first 100 steps to print.')
subprocess.run(cmd)
