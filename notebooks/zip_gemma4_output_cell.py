from pathlib import Path
import shutil

output_dir = Path("/content/gemma4-e4b-unsloth-phishing")
if not output_dir.exists():
    output_dir = Path("gemma4-e4b-unsloth-phishing")
if not output_dir.exists():
    output_dir = Path("runs/gemma4-e4b-unsloth-phishing")
if not output_dir.exists():
    raise FileNotFoundError("Could not find gemma4-e4b-unsloth-phishing output folder.")

zip_base = output_dir.parent / output_dir.name
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
print(f"Created: {zip_path}")

try:
    from google.colab import files

    files.download(zip_path)
except ImportError:
    pass
