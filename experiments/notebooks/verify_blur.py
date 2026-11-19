import pandas as pd
import cv2
import numpy as np

df = pd.read_csv("reports/train_blur_report.csv")

# ----- Show 5 blurries vs 5 sharp images side by side -----

blurry = df.nsmallest(8, "blur_score")
sharp = df.nlargest(8, "blur_score")

for (_, br), (_, sr) in zip(blurry.iterrows(), sharp.iterrows()):
    img_b = cv2.imread(br["file_path"])
    img_s = cv2.imread(sr["file_path"])

    if img_b is None or img_s is None :
        continue

    img_b = cv2.resize(img_b, (224,224))
    img_s = cv2.resize(img_s, (224,224))

    pair = np.hstack([img_b, img_s])
    cv2.putText(pair, f"Blurry:{br['blur_score']:.2f}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(pair, f"Sharp:{sr['blur_score']:.2f}", (235,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow("Blurry(left) vs Sharp(right)", pair)
    cv2.waitKey(0)

cv2.destroyAllWindows()
