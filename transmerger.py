import pandas as pd
import glob

files = sorted(glob.glob("translated*.csv"))

print(files)  # Check which files will be merged

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df.to_csv("Hindi_Dataset.csv", index=False, encoding="utf-8-sig")

print("Merged Successfully!")
print("Total rows:", len(df))