About Dataset
FallAllD/Derived FallAllD Dataset - Fall Detection

1. Original FallAllD.pkl by Majd SALEH

Description:
FallAllD is a large open dataset of human falls and activities of daily living simulated by 15 participants. FallAllD consists of 26420 files collected using three data-loggers worn on the waist, wrist and neck of the subjects. Motion signals are captured using an accelerometer, gyroscope, magnetometer and barometer with efficient configurations that suit the potential applications e.g. fall detection, fall prevention and human activity recognition.

FallAllD is described in detail in the following journal article:

M. Saleh, M. Abbas and R. L. B. Jeannès, "FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications," in IEEE Sensors Journal, doi: 10.1109/JSEN.2020.3018335.

Attributes:

• Sensors: Accelerometer (Acc), Gyroscope (Gyr), Magnetometer (Mag), Barometer (Bar)
• Device Positions: Waist, Wrist, Neck
• Sampling Rate: 238 Hz/80 Hz/10 Hz
• Activities: Multiple types of falls and various ADLs
• Data Format: Pickle file containing raw sensor data with corresponding activity labels

2. activity_info.pkl

Description:
The activity_info.pkl file is a derived dataset that maps activity IDs to their corresponding descriptions. It serves as a reference to understand the different activities included in the FallAllD dataset and the new derived dataset.

Attributes:

• ActivityID: Unique identifier for each activity
• Description: Text description of the activity

Purpose:

• To provide a clear understanding of the activities represented by the ActivityIDs in the datasets.
• Facilitates interpretation of activity labels during data analysis and model evaluation.

3. FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl

Description:
The FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl is a processed and refined version of the original FallAllD dataset. It has been modified to focus on specific sensor data and activity types, making it more suitable for multi-class fall detection using machine learning.

Modifications:

• Downsampling to 40Hz: The original data, sampled at 238Hz, was downsampled to 40Hz to reduce the dataset size and computational requirements.
• Removing 'Mag' and 'Bar' Sensor Data: Magnetometer (Mag) and Barometer (Bar) sensor data were removed to simplify the dataset, focusing only on accelerometer (Acc) and gyroscope (Gyr) data.
• Removing Unnecessary 'ActivityIDs': Activities that were not relevant to the study or had insufficient data were removed to streamline the dataset.
• Balancing the Classes with SMOTE: Synthetic Minority Over-sampling Technique (SMOTE) was applied to balance the classes, addressing the issue of imbalanced data and ensuring a more robust model training process.
• Removing 'Neck' Device Data: Data from the neck device was removed to focus on sensor data from the waist device, which was deemed more relevant for this study.

Attributes:

• Sensors: Accelerometer (Acc), Gyroscope (Gyr)
• Device Position: Waist, Wrist
• Sampling Rate: 40 samples per second
• Activities: Filtered set of fall types and ADLs, represented by a refined list of ActivityIDs
• Data Format: Pickle file containing processed sensor data with corresponding activity labels

The Original Dataset was taken from IEEE Paper, "FallAllD: A Comprehensive Dataset of Human Falls and Activities of Daily Living".
M. Saleh, M. Abbas and R. L. B. Jeannès, "FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications," in IEEE Sensors Journal, doi: 10.1109/JSEN.2020.3018335.