import os
from pathlib import Path

root = Path("C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed")   # change if needed

for writer_folder in sorted(root.iterdir()):
    if writer_folder.is_dir():
        writer_id = writer_folder.name  # e.g., HAMS_A
        counter = 1
        
        for img in sorted(writer_folder.iterdir()):
            if img.is_file():
                # keep extension
                ext = img.suffix.lower()

                # extract type if present (C/W/S)
                """
                parts = img.stem.split("-")
                sample_type = None
                for p in parts:
                    if p in {"C", "W", "S"}:
                        sample_type = p
                        break
                """

                if sample_type:
                    new_name = f"{writer_id}_{sample_type}_{counter:03d}{ext}"
                else:
                    new_name = f"{writer_id}_{counter:03d}{ext}"

                img.rename(writer_folder / new_name)
                counter += 1

print("Done — files renamed safely.")
