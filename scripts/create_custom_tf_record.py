import os
import io
import hashlib
import tensorflow as tf
import xml.etree.ElementTree as ET

LABELS = {
    "hello": 1,
    "thanks": 2,
    "yes": 3,
    "no": 4,
}

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_record(folder, output):
    writer = tf.io.TFRecordWriter(output)
    xml_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".xml")
    )

    print("Found XML files:", len(xml_files))

    for xml_name in xml_files:
        xml_path = os.path.join(folder, xml_name)
        root = ET.parse(xml_path).getroot()

        filename = root.findtext("filename")
        image_path = os.path.join(folder, filename)

        if not os.path.exists(image_path):
            print("Missing image:", filename)
            continue

        with tf.io.gfile.GFile(image_path, "rb") as fid:
            encoded = fid.read()

        image = tf.io.decode_image(encoded, channels=3)
        height = int(image.shape[0])
        width = int(image.shape[1])

        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes_text, classes = [], []

        for obj in root.findall("object"):
            name = obj.findtext("name").strip()

            if name not in LABELS:
                print("Unknown label:", name)
                continue

            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin")) / width
            xmax = float(box.findtext("xmax")) / width
            ymin = float(box.findtext("ymin")) / height
            ymax = float(box.findtext("ymax")) / height

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append(name.encode("utf8"))
            classes.append(LABELS[name])

        if not classes:
            print("No valid objects:", xml_name)
            continue

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": int64_feature(height),
                    "image/width": int64_feature(width),
                    "image/filename": bytes_feature(filename.encode("utf8")),
                    "image/source_id": bytes_feature(filename.encode("utf8")),
                    "image/encoded": bytes_feature(encoded),
                    "image/format": bytes_feature(b"jpeg"),
                    "image/object/bbox/xmin": float_list_feature(xmins),
                    "image/object/bbox/xmax": float_list_feature(xmaxs),
                    "image/object/bbox/ymin": float_list_feature(ymins),
                    "image/object/bbox/ymax": float_list_feature(ymaxs),
                    "image/object/class/text": bytes_list_feature(classes_text),
                    "image/object/class/label": int64_list_feature(classes),
                }
            )
        )

        writer.write(example.SerializeToString())

    writer.close()
    print("Created:", output)

if __name__ == "__main__":
    os.makedirs("Tensorflow/workspace/annotations", exist_ok=True)

    create_record(
        "Tensorflow/workspace/images/train_final",
        "Tensorflow/workspace/annotations/train.record",
    )

    create_record(
        "Tensorflow/workspace/images/test_final",
        "Tensorflow/workspace/annotations/test.record",
    )
