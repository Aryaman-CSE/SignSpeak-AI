from pathlib import Path

path = Path("realtime_signspeak.py")
source = path.read_text(encoding="utf-8-sig")

old = '''    model_path = ROOT / "models" / "asl_mobilenetv2" / "webcam_finetuned_model.h5"
    class_path = ROOT / "models" / "asl_mobilenetv2" / "class_indices.json"
'''

new = '''    model_path = SCRIPT_DIR / "models" / "asl_mobilenetv2" / "webcam_finetuned_model.h5"
    class_path = SCRIPT_DIR / "models" / "asl_mobilenetv2" / "class_indices.json"
'''

if old not in source:
    raise RuntimeError("Expected model path lines not found. No changes made.")

source = source.replace(old, new, 1)
compile(source, str(path), "exec")
path.write_text(source, encoding="utf-8")

print("[SUCCESS] Model paths corrected.")
