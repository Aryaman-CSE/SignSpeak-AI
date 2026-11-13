import os
base = 'SignSpeak-AI-main' if os.path.exists('SignSpeak-AI-main/Tensorflow') else '.'
lbl_path = os.path.join(base, 'Tensorflow/workspace/annotations/label_map.pbtxt')

letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
content = "\n".join([f"item {{\n\tname:'{l}'\n\tid:{idx+1}\n}}" for idx, l in enumerate(letters)])

with open(lbl_path, 'w', encoding='utf-8') as f:
    f.write(content + "\n")

print("Updated label_map.pbtxt with 26 classes (A-Z)!")
