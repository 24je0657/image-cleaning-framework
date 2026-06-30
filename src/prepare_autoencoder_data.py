import pandas as pd

REPORTS_DIR = "reports"

def prepare_clean_subset(split = "train", samples_perclass = 400):
    blur_df = pd.read_csv(f"{REPORTS_DIR}/{split}_blur_report.csv")
    dup_df = pd.read_csv(f"{REPORTS_DIR}/{split}_duplicates_report.csv")

    # ----- Keep Only Images whch are Sharp and Non Duplicates -----
    merged = blur_df.merge(
        dup_df[["file_path","Exact_duplicates","near_duplicate"]],
        on = "file_path",  how = "left"
    )
    clean = merged[(merged["is_blurry"] == False) & 
                   (merged["Exact_duplicates"].fillna(False) == False) &
                   (merged["near_duplicate"].fillna(False) == False)
                   ]
    
    print(f" Clean Images : {len(clean)} / {len(merged)}")

    sampled_frames = []
    for class_name in clean["class"].unique():
        class_subset = clean[clean["class"] == class_name]
        n = min(samples_perclass, len(class_subset))
        sampled = class_subset.sample(n=n, random_state = 42)
        sampled_frames.append(sampled)

        print(f" {class_name} : {n} Images selected")
    
    clean_final = pd.concat(sampled_frames, ignore_index = True)
    
    # ----- Save just the file path + class -----

    out_path = f"{REPORTS_DIR}/clean_{split}.csv"
    clean_final[["file_path", "class"]].to_csv(
        out_path,
        index = False
    )

    print(f" \n Clean Subset Reference saved => {out_path}")
    print(f" Total Images Referenced {len(clean_final)}")

    return clean_final

if __name__ == "__main__":
    prepare_clean_subset("train")






