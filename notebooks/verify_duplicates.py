import pandas as pd
import cv2
import numpy as np

df = pd.read_csv("reports/train_duplicates_report.csv")
print(df.columns)

# ----- Pick 3 near duplicate images and inspect -----
near_dup = df[df["near_duplicate"] == True].head(3)

for idx, row in near_dup.iterrows():
    img1 = cv2.imread(row["file_path"])
    img2 = cv2.imread(row["near_duplicate_of"])

    if img1 is None or img2 is None:
        continue

    img1 = cv2.resize(img1,(224,224))
    img2 = cv2.resize(img2,(224,224))

    pair = np.hstack([img1,img2])
    label = f" score:{row['cosine_score']}"
    cv2.putText(pair, label, (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow(f" Near Dup Pair {idx}", pair)

cv2.waitKey(0)
cv2.destroyAllWindows()




