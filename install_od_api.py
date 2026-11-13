import os, urllib.request, zipfile, subprocess, sys, glob

py = sys.executable
base = os.path.abspath('SignSpeak-AI-main' if os.path.exists('SignSpeak-AI-main/Tensorflow') else '.')
tf_dir = os.path.join(base, 'Tensorflow')
models_dir = os.path.join(tf_dir, 'models')
protoc_dir = os.path.join(tf_dir, 'protoc')

# 1. Clone TensorFlow Models repo if not present
if not os.path.exists(os.path.join(models_dir, 'research', 'object_detection')):
    print('[INFO] Cloning TensorFlow Object Detection API repository...')
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/tensorflow/models', models_dir], check=True)

# 2. Download Windows Protoc 3.19.6
os.makedirs(protoc_dir, exist_ok=True)
protoc_exe = os.path.join(protoc_dir, 'bin', 'protoc.exe')
if not os.path.exists(protoc_exe):
    print('[INFO] Downloading protoc-3.19.6 for Windows...')
    url = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-win64.zip'
    zip_path = os.path.join(protoc_dir, 'protoc.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(protoc_dir)

# 3. Compile .proto files
research_dir = os.path.join(models_dir, 'research')
print('[INFO] Compiling protobuf files...')
proto_files = glob.glob(os.path.join(research_dir, 'object_detection', 'protos', '*.proto'))
subprocess.run([protoc_exe, f'--python_out={research_dir}'] + [os.path.relpath(p, research_dir) for p in proto_files], cwd=research_dir, check=True)

# 4. Install required lightweight dependencies for TF 2.10.1 Object Detection
print('[INFO] Installing Object Detection dependencies into Python 3.10...')
deps = [
    'tf_slim', 'cython', 'matplotlib', 'scipy', 'six', 'gin-config',
    'lvis', 'pycocotools-windows; platform_system=="Windows"', 'tf-models-official==2.10.0',
    'tensorflow-io==0.27.0', 'keras==2.10.0', 'tensorflow==2.10.1', 'protobuf==3.19.6', 'numpy<2.0'
]
subprocess.run([py, '-m', 'pip', 'install'] + deps)

# 5. Add research and slim to Python's site-packages via a .pth file (cleaner than setup.py!)
import site
site_pkg = site.getusersitepackages() if not site.getsitepackages() else [p for p in site.getsitepackages() if 'site-packages' in p][0]
pth_file = os.path.join(site_pkg, 'tf_object_detection.pth')
with open(pth_file, 'w') as f:
    f.write(research_dir + '\n')
    f.write(os.path.join(research_dir, 'slim') + '\n')

print('[SUCCESS] TensorFlow Object Detection API installed and linked via:', pth_file)
