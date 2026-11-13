import os, glob, io, re, xml.etree.ElementTree as ET
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image

base = 'SignSpeak-AI-main' if os.path.exists('SignSpeak-AI-main/Tensorflow') else '.'
lbl_txt = open(os.path.join(base, 'Tensorflow/workspace/annotations/label_map.pbtxt')).read()
label_map = {m.group(1): int(m.group(2)) for m in re.finditer(r"name\s*:\s*['\"]([^'\"]+)['\"][\s\S]*?id\s*:\s*(\d+)", lbl_txt)}
print('Loaded classes:', len(label_map))

def make_rec(img_dir, out_file):
    w = tf.io.TFRecordWriter(out_file)
    cnt = 0
    for x in glob.glob(os.path.join(img_dir, '*.xml')):
        t = ET.parse(x).getroot()
        fn = t.find('filename').text
        p = os.path.join(img_dir, fn)
        if not os.path.exists(p):
            p = os.path.splitext(x)[0] + '.jpg'
        raw = open(p, 'rb').read()
        im = Image.open(io.BytesIO(raw))
        width, height = im.size
        xmins, xmaxs, ymins, ymaxs, txt, cls = [], [], [], [], [], []
        for o in t.findall('object'):
            n = o.find('name').text
            if n in label_map:
                b = o.find('bndbox')
                xmins.append(float(b.find('xmin').text) / width)
                xmaxs.append(float(b.find('xmax').text) / width)
                ymins.append(float(b.find('ymin').text) / height)
                ymaxs.append(float(b.find('ymax').text) / height)
                txt.append(n.encode('utf8'))
                cls.append(label_map[n])
        ex = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fn.encode('utf8')])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fn.encode('utf8')])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=txt)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=cls)),
        }))
        w.write(ex.SerializeToString())
        cnt += 1
    w.close()
    print(f'Created {out_file} with {cnt} records!')

make_rec(os.path.join(base, 'Tensorflow/workspace/images/train'), os.path.join(base, 'Tensorflow/workspace/annotations/train.record'))
make_rec(os.path.join(base, 'Tensorflow/workspace/images/test'), os.path.join(base, 'Tensorflow/workspace/annotations/test.record'))
