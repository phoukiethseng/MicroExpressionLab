import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2 as cv

DATASET_PATH = "F:\\CASME2\\Cropped" # Path to dataset containing cropped images of CASMEII
OUTPUT_DATASET_PATH = "D:\\test"
ORIGINAL_LABEL_FILE_PATH = "F:\\CASME2\\CASME2-coding-20140508 - V2.xlsx"

def frame_path(subject_number, expression_id, frame_number, dataset_path=DATASET_PATH):
    return os.path.join(dataset_path, 'sub{:02d}'.format(subject_number), expression_id, f'reg_img{frame_number}.jpg')


objective_class = pd.read_excel(
    "F:\\CASME2\\CASME2-ObjectiveClasses.xlsx")  # This contains label using objective class 1-7
frame_info = pd.read_excel(ORIGINAL_LABEL_FILE_PATH, usecols=(0, 1, 3, 4, 5),
                           dtype={'OnsetFrame': 'Int32', 'ApexFrame': 'Int32', 'OffsetFrame': 'Int32',
                                  'Subject': 'Int32'}, na_values={'ApexFrame': '/'}).dropna()

objective_class.set_index(['Subject', 'Filename'])
frame_info.set_index(['Subject', 'Filename'])

label = pd.merge(frame_info, objective_class, on=['Subject', 'Filename'], how='inner')

label.insert(3, 'OnsetApexFrame', ((label.OnsetFrame.to_numpy() + label.ApexFrame.to_numpy()) / 2).astype(np.int32))
label.insert(5, 'ApexOffsetFrame', ((label.ApexFrame.to_numpy() + label.OffsetFrame.to_numpy()) / 2).astype(np.int32))

#TODO: Make sure every frame file reference in label ACTUALLY EXIST before we do computationally expensive augmentation

horizontal_flips = [False, True]
translations = [(0, 0), (-2, -2), (-2, 2), (2, -2), (2, 2)]
rotations = [-10, -5, 0, 5, -10]
scales = [0.9, 1.0, 1.1]

transformers = []
for horizontal_flip in horizontal_flips:
    for translation in translations:
        for rotation in rotations:
            for scale in scales:
                transformer = A.Compose([
                    A.HorizontalFlip(p=1.0 if horizontal_flip else 0.0),
                    A.Affine(
                        p=1.0,
                        translate_px={'x': translation[0], 'y': translation[1]},
                        scale=scale,
                        rotate=rotation,
                    )
                ])
                transformers.append(transformer)


def augment_and_save(image, current_index, exp_class, exp_state, output_path=OUTPUT_DATASET_PATH):
    c = current_index
    s =  len(transformers)
    label = {'Filename': [], 'ExpressionClass': [exp_class for _ in range(s)], 'ExpressionState': [exp_state for _ in range(s)]}
    for transformer in transformers:
        augmented_image = transformer(image=image)['image']
        filename = "{:09d}.jpg".format(c)
        absolute_file_path = os.path.join(OUTPUT_DATASET_PATH, filename)
        cv.imwrite(absolute_file_path, augmented_image)
        c += 1
        label['Filename'].append(filename)

    pd.DataFrame(label).to_csv(os.path.join(output_path, "label.csv"), mode='a',index=False, columns=['Filename', 'ExpressionClass', 'ExpressionState'], header=False)
    return c


index = 1
for _, row in label.iterrows():
    exp_class = row['Objective Class']

    onset_frame = cv.imread(
        frame_path(subject_number=row['Subject'], expression_id=row['Filename'], frame_number=row['OnsetFrame']))
    onsetapex_frame = cv.imread(
        frame_path(subject_number=row['Subject'], expression_id=row['Filename'], frame_number=row['OnsetApexFrame']))
    apex_frame = cv.imread(
        frame_path(subject_number=row['Subject'], expression_id=row['Filename'], frame_number=row['ApexFrame']))
    apexoffset_frame = cv.imread(
        frame_path(subject_number=row['Subject'], expression_id=row['Filename'], frame_number=row['ApexOffsetFrame']))
    offset_frame = cv.imread(
        frame_path(subject_number=row['Subject'], expression_id=row['Filename'], frame_number=row['OffsetFrame']))

    index = augment_and_save(onset_frame, exp_class=exp_class, exp_state=1, current_index=index)
    index = augment_and_save(onsetapex_frame, exp_class=exp_class, exp_state=2, current_index=index)
    index = augment_and_save(apex_frame, exp_class=exp_class, exp_state=3, current_index=index)
    index = augment_and_save(apexoffset_frame, exp_class=exp_class, exp_state=4, current_index=index)
    index = augment_and_save(offset_frame, exp_class=exp_class, exp_state=5, current_index=index)

    print(f'Subject {row['Subject']} Expression ID {row['Filename']}')

# pd.DataFrame(processed_label).to_excel(os.path.join(OUTPUT_DATASET_PATH, "label.xlsx"))
