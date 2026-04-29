6th Semester Mini Project - Injection Molding Quality Assessment

Data
- Tabular data: [data/dataset.csv](data/dataset.csv) - 564 samples, 334 features from 9 data sources; use `MachineCycleID` to link with images
- Data variables & labels: [data/data_description.pdf](data/data_description.pdf)
- Computer-vision images [6.9GB]: [data/Rohbilder](data/Rohbilder) - raw surface images per cycle
- Thermal images [1.2GB]: [data/Thermographie](data/Thermographie) - infrared images per cycle

Note: CV & Thermal images are not pushed in the repo as they were obtained through requested access.

Dataset info
- 564 injection-molded parts from 47 experimental points (12 parts each)
- 2 materials: PP and ABS (70% recyclate)
- Manually labeled quality (OK / NOK and sublabels)
- Design of Experiment: 4 parameters varied (cylinder temp, mold temp, injection speed, holding pressure)

Source
- Original dataset: https://b2share.eudat.eu/records/k0v7s-jf859
- Project: ProBayes (SKZ / Fraunhofer IPA, 2021/2022)
- Converted from Parquet (`dataset_v2.parquet`) to CSV


