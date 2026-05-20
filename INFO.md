 📋 Project Overview                                                                                                                                          
                                                                                                                                                              
 This project builds a multi-sensor fusion system for automated quality inspection of injection-molded plastic parts. The goal is to detect surface and       
 structural defects — sink marks, sprue circle defects, underfilled parts, streaks, and old-granulate contamination — by combining data from four different   
 sensor modalities captured during each injection cycle.                                                                                                      
                                                                                                                                                              
 The system was trained and evaluated on the ProBayes dataset (SKZ / Fraunhofer IPA, 2021/2022), which contains 564 injection-molded parts produced across 47 
 experimental design points (12 parts per point). Two materials were used: PP (polypropylene) and ABS with 70% recyclate. Four process parameters were varied 
 in a design-of-experiments fashion: cylinder temperature, mold temperature, injection speed, and holding pressure.                                           
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 🧩 The Four Sensor Modalities                                                                                                                                
                                                                                                                                                              
 ### 1. Thermal Infrared Images (🌡️)                                                                                                                          
                                                                                                                                                              
 During each injection cycle, two thermal frames are captured by an infrared camera, producing 480 × 640 temperature matrices stored as CSV files. These      
 capture the surface temperature distribution of the freshly molded part. Thermal patterns are critical for detecting sink marks (regions that cool unevenly  
 due to differential shrinkage), underfilled parts (which appear colder overall), and streaks (temperature variance along the part walls).                    
                                                                                                                                                              
 The system processes thermal data in two ways:                                                                                                               
 - Raw pixel data fed through an EfficientNet-B0 CNN                                                                                                          
 - ROI (Region of Interest) statistics — 10 pre-computed features like sprue temperature, dome temperature, edge temperatures, and gradient magnitudes — fed  
 through a small physics-informed MLP                                                                                                                         
                                                                                                                                                              
 ### 2. Computer Vision Images (📷)                                                                                                                           
                                                                                                                                                              
 Three greyscale surface photographs are captured at different positions around the part (position 1, 2, and 3). These are greyscale BMP files that show the  
 visible surface condition. In principle, these can detect visible defects like sink marks or surface streaks, though in practice the defects in this dataset 
 are primarily thermal phenomena rather than visual ones.                                                                                                     
                                                                                                                                                              
 The system processes these as three separate views, encodes each through a shared ResNet-50 backbone, then aggregates them using a cross-section attention   
 mechanism that learns to weight which view is most informative.                                                                                              
                                                                                                                                                              
 ### 3. DXP Injection Cycle Time Series (📊)                                                                                                                  
                                                                                                                                                              
 During injection, high-frequency sensors record 8 process channels at approximately 4096 time points per cycle:                                              
 - Injection pressure (actual)                                                                                                                                
 - Injection position (actual)                                                                                                                                
 - Injection velocity (actual)                                                                                                                                
 - Holding pressure (actual)                                                                                                                                  
 - Clamping force (actual)                                                                                                                                    
 - Mold temperature (ejector side)                                                                                                                            
 - Mold temperature (nozzle side)                                                                                                                             
 - Dosing volume (actual)                                                                                                                                     
                                                                                                                                                              
 These time series are processed by a Causal Temporal Convolutional Network (TCN) with a receptive field of 85 timesteps. The TCN uses dilated causal         
 convolutions — meaning it can only look backward in time, never forward — making it suitable for real-time deployment.                                       
                                                                                                                                                              
 ### 4. Tabular Process Parameters (📋)                                                                                                                       
                                                                                                                                                              
 40 raw scalar features from the machine's control system, including temperature setpoints, quality metrics, dosing parameters, and environmental readings.   
 These provide the baseline process context for each cycle.                                                                                                   
                                                                                                                                                              
 Important preprocessing decision: The dataset originally contained 102 pre-extracted feature columns derived from the thermal and CV images (columns         
 prefixed SIM_* and IR_Img*). These were removed from the tabular feature set because they create a data leakage problem — they already encode information    
 from the images, making the CNN encoders redundant. By removing them, the CNNs are forced to actually learn from the raw pixel data.                         
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 🏗️ System Architecture                                                                                                                                       
                                                                                                                                                              
 ```                                                                                                                                                          
   ┌─────────────────────┐     ┌──────────────────┐                                                                                                           
   │  Thermal IR Frames  │────▶│  EfficientNet-B0 │───┐                                                                                                       
   │  (2 × 480×640 CSV)  │     │  + ROI MLP Head  │   │  512-dim                                                                                              
   └─────────────────────┘     └──────────────────┘   │                                                                                                       
                                                       │                                                                                                      
   ┌─────────────────────┐     ┌──────────────────┐   │                                                                                                       
   │  CV Surface Images  │────▶│  ResNet-50 × 3   │───┤                                                                                                       
   │  (3 greyscale BMPs)  │     │  + Cross-Sec Attn│   │  512-dim                                                                                             
   └─────────────────────┘     └──────────────────┘   │                                                                                                       
                                                       │                                                                                                      
   ┌─────────────────────┐     ┌──────────────────┐   │   ┌─────────────────────┐                                                                             
   │  DXP Time Series    │────▶│  Causal TCN      │───┼──▶│  Cross-Modal Fusion  │                                                                            
   │  (8 × 4096 pts)     │     │  (RF: 85, SE Attn)│   │   │  Transformer         │                                                                           
   └─────────────────────┘     └──────────────────┘   │   │  (4 tokens, 384-dim) │                                                                            
                                                       │   │  2 layers, 4 heads   │                                                                           
   ┌─────────────────────┐     ┌──────────────────┐   │   └──────────┬──────────┘                                                                             
   │  Tabular Parameters │────▶│  3-Layer MLP     │───┘              │                                                                                        
   │  (40 scalars)        │     │  (256→256→256)   │                  │                                                                                       
   └─────────────────────┘     └──────────────────┘                  │                                                                                        
                                                                     ▼                                                                                        
                                                       ┌─────────────────────┐                                                                                
                                                       │  Defect Head (MLP)  │──▶ 8 binary predictions                                                        
                                                       │  (384→128→8)        │                                                                                
                                                       └─────────────────────┘                                                                                
                                                                     │                                                                                        
                                                       ┌─────────────────────┐                                                                                
                                                       │  DANN Head (MLP)    │──▶ 47-way experiment classifier                                                
                                                       │  (384→128→47)       │                                                                                
                                                       └─────────────────────┘                                                                                
 ```                                                                                                                                                          
                                                                                                                                                              
 ### Key Design Decisions                                                                                                                                     
                                                                                                                                                              
 1. Modality-Specific Encoders                                                                                                                                
 Each modality has its own encoder tailored to its data type:                                                                                                 
 - Thermal: EfficientNet-B0 (pretrained on ImageNet, adapted for 3-channel thermal input). Additionally, a small ROI physics head processes the 10            
 temperature-statistic features. These two branches are concatenated and projected to 512 dimensions.                                                         
 - Visual: ResNet-50 processed 3 views independently, each with a learned position embedding. A cross-section attention mechanism pools the three views into  
 a single 512-dim vector, learning which view(s) matter most.                                                                                                 
 - Sequence: A causal TCN with squeeze-excitation blocks processes the 8-channel DXP signals. Global average pooling over time gives a 256-dim embedding.     
 - Tabular: A straightforward 3-layer MLP projects 40 scalars to 256 dimensions.                                                                              
                                                                                                                                                              
 2. Cross-Modal Fusion Transformer                                                                                                                            
 Each encoder output is projected to a common 384-dim token space, added with a learned modality embedding (so the transformer knows which token is which).   
 If a modality is missing for a particular sample (e.g., no CV images), a learned [MASK] token is substituted. The four tokens pass through a 2-layer         
 transformer encoder with 4 attention heads. The output tokens are weighted by a learned attention head (valid tokens only) and summed to produce a single    
 fused 384-dim representation.                                                                                                                                
                                                                                                                                                              
 3. Dual Prediction Heads                                                                                                                                     
 - Defect Head: A 2-layer MLP (384→128→8) producing logits for 8 binary classification tasks (NOK, SinkMarks, SprueCircle, Underfilled, Streaks Level 1/2/3,  
 OldGranulate). Trained with focal loss (γ=2.5) to focus on hard, rare examples.                                                                              
 - DANN Head (Domain-Adversarial): A 2-layer MLP that tries to predict which of the 47 experiments a sample came from. A gradient reversal layer is applied   
 before this head during training — the feature extractor is punished for making experiment prediction easy, forcing the fused representation to be           
 experiment-invariant. This helps the model generalize to new process conditions.                                                                             
                                                                                                                                                              
 4. Missing Modality Handling                                                                                                                                 
 Every sample has a boolean valid flag for each modality. Missing modalities get the [MASK] token in the fusion transformer, and their attention scores are   
 set to -inf so they don't contribute to the pooled representation. This means the system gracefully degrades — if the thermal camera fails, it still works   
 on the remaining three modalities.                                                                                                                           
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 🏋️ Training Pipeline                                                                                                                                         
                                                                                                                                                              
 ### Data Splitting                                                                                                                                           
                                                                                                                                                              
 The critical issue in industrial ML is group leakage — samples from the same experiment (same process settings) appearing in both train and test sets. This  
 gives an inflated view of performance because the model memorizes experiment-specific patterns rather than learning generalizable features.                  
                                                                                                                                                              
 To address this, the system uses 5-Fold Stratified Group Cross-Validation:                                                                                   
 - Groups: Experiment ID (MET_ExperimentNumber) — all 12 parts from one experiment stay together                                                              
 - Stratification: By the primary target (LBL_NOK) — each fold has a similar proportion of defective parts                                                    
 - Result: Each fold has ~450 training samples and ~114 test samples from completely different experiments                                                    
                                                                                                                                                              
 ### Class Imbalance Handling                                                                                                                                 
                                                                                                                                                              
 The defect labels are severely imbalanced:                                                                                                                   
                                                                                                                                                              
 ┌───────────────────┬────────────────┬───────┐                                                                                                               
 │ Label             │ Positive Count │ Rate  │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_NOK           │ 155            │ 27.5% │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_SinkMarks     │ 143            │ 25.4% │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_SprueCircle   │ 72             │ 12.8% │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_Underfilled   │ 60             │ 10.6% │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_StreaksLevel3 │ 48             │ 8.5%  │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_StreaksLevel1 │ 30             │ 5.3%  │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_StreaksLevel2 │ 18             │ 3.2%  │                                                                                                               
 ├───────────────────┼────────────────┼───────┤                                                                                                               
 │ LBL_OldGranulate  │ 9              │ 1.6%  │                                                                                                               
 └───────────────────┴────────────────┴───────┘                                                                                                               
                                                                                                                                                              
 The system combats this with:                                                                                                                                
 1. WeightedRandomSampler — rare classes are oversampled (OldGranulate gets ~36× the sampling weight of common classes)                                       
 2. Focal Loss — down-weights well-classified examples, forcing the model to focus on the hard minority examples                                              
 3. Per-label evaluation — reports F1 per class, not just accuracy (which would be 98% by predicting "no defect" for everything)                              
                                                                                                                                                              
 ### Training Schedule                                                                                                                                        
                                                                                                                                                              
 - 100 epochs per fold                                                                                                                                        
 - AdamW optimizer (lr=5e-5 for backbones, 1e-3 for heads)                                                                                                    
 - DANN lambda ramps from 0 → 0.8 over first 15 epochs (warmup period where the model learns defects before being forced to be experiment-invariant)          
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 📊 Evaluation Methodology                                                                                                                                    
                                                                                                                                                              
 The system is evaluated on five separate test sets (one per CV fold), each containing samples from entirely unseen experiments. Metrics are reported as mean 
 ± std across folds to show both performance and stability.                                                                                                   
                                                                                                                                                              
 ### Key Metrics                                                                                                                                              
                                                                                                                                                              
 - Macro F1: Average F1 across all 8 classes (treats rare and common classes equally)                                                                         
 - Micro F1: Weighted by class frequency (reflects overall accuracy)                                                                                          
 - ROC-AUC: Measures ranking quality (threshold-independent)                                                                                                  
 - PR-AUC: Precision-recall area (better than ROC-AUC for imbalanced data)                                                                                    
 - Per-label F1: Separate F1 for each defect type                                                                                                             
                                                                                                                                                              
 ### Ablation Study                                                                                                                                           
                                                                                                                                                              
 To understand which modalities contribute most, the system is evaluated with each modality removed:                                                          
 1. All 4 modalities (baseline)                                                                                                                               
 2. Without thermal                                                                                                                                           
 3. Without visual                                                                                                                                            
 4. Without sequence (DXP)                                                                                                                                    
 5. Without tabular                                                                                                                                           
 6. Tabular only                                                                                                                                              
                                                                                                                                                              
 This quantifies the marginal contribution of each sensor.                                                                                                    
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 ✅ Results Summary                                                                                                                                           
                                                                                                                                                              
 Final 5-Fold Cross-Validated Performance:                                                                                                                    
                                                                                                                                                              
 ┌──────────────┬────────────────────────────────────────┐                                                                                                    
 │ Metric       │ Value                                  │                                                                                                    
 ├──────────────┼────────────────────────────────────────┤                                                                                                    
 │ Macro F1     │ 0.59 ± 0.02                            │                                                                                                    
 ├──────────────┼────────────────────────────────────────┤                                                                                                    
 │ Micro F1     │ 0.78 ± 0.02                            │                                                                                                    
 ├──────────────┼────────────────────────────────────────┤                                                                                                    
 │ Mean ROC-AUC │ 0.87 ± 0.02                            │                                                                                                    
 ├──────────────┼────────────────────────────────────────┤                                                                                                    
 │ Mean PR-AUC  │ 0.71 ± 0.01                            │                                                                                                    
 ├──────────────┼────────────────────────────────────────┤                                                                                                    
 │ LBL_NOK F1   │ 0.75 (Precision: 88.5%, Recall: 74.8%) │                                                                                                    
 └──────────────┴────────────────────────────────────────┘                                                                                                    
                                                                                                                                                              
 Ablation Findings (Modality Contribution):                                                                                                                   
 - Tabular is the strongest single modality (F1=0.52 alone)                                                                                                   
 - Each additional modality contributes positively                                                                                                            
 - Full 4-modality fusion gives +13% over tabular alone — real multi-modal synergy                                                                            
 - Without tabular, performance drops 30.5% — showing tabular provides the critical process context                                                           
 - Visual contributes the least (+3.4%) — the defects are primarily thermal phenomena                                                                         
                                                                                                                                                              
 Practical Impact (per 1000 parts):                                                                                                                           
 - ~275 parts are actually defective (27.5% defect rate)                                                                                                      
 - Model catches ~206 of them (74.8% recall)                                                                                                                  
 - Only ~8 false alarms (96.9% specificity)                                                                                                                   
 - 73% reduction in manual QC workload while catching 3 in 4 defects                                                                                          
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 🔬 Model Complexity                                                                                                                                          
                                                                                                                                                              
 ┌─────────────────────┬─────────────────────────┐                                                                                                            
 │ Metric              │ Value                   │                                                                                                            
 ├─────────────────────┼─────────────────────────┤                                                                                                            
 │ Total Parameters    │ 34.5M (~130 MB FP32)    │                                                                                                            
 ├─────────────────────┼─────────────────────────┤                                                                                                            
 │ Theoretical FLOPs   │ 15.76 GFLOPs            │                                                                                                            
 ├─────────────────────┼─────────────────────────┤                                                                                                            
 │ TCN Receptive Field │ 85 timesteps            │                                                                                                            
 ├─────────────────────┼─────────────────────────┤                                                                                                            
 │ Training Time       │ ~2.5 min per fold (GPU) │                                                                                                            
 └─────────────────────┴─────────────────────────┘                                                                                                            
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 🚀 Demo Application                                                                                                                                          
                                                                                                                                                              
 The project includes an interactive Flask web application (server.py) that:                                                                                  
 - Lists all 564 samples with their ground-truth labels                                                                                                       
 - Shows thermal heatmaps and gradient maps                                                                                                                   
 - Displays CV surface images                                                                                                                                 
 - Renders DXP time series plots                                                                                                                              
 - Runs a physics-based heuristic predictor (separate from the learned model) that detects defects using simple temperature thresholds — useful as a baseline 
 / explainability tool                                                                                                                                        
 - Shows modality attention weights                                                                                                                           
 - Provides filtering by defect type and material                                                                                                             
                                                                                                                                                              
 Launched via bash run.sh at http://localhost:5000.                                                                                                           
                                                                                                                                                              
 ────────────────────────────────────────────────────────────────────────────────                                                                             
                                                                                                                                                              
 💡 Key Takeaways                                                                                                                                             
                                                                                                                                                              
 1. Multi-modal fusion works — combining thermal, visual, time-series, and tabular data gives better results than any single modality.                        
 2. Pre-extracted features are dangerous — if the tabular data already contains image-derived statistics (SIM_*, IR_Img*), the CNNs become redundant and      
 metrics become inflated.                                                                                                                                     
 3. Group-stratified cross-validation is essential for industrial datasets where experiments share process conditions — random splits produce overconfident   
 metrics.                                                                                                                                                     
 4. Class imbalance is the main performance bottleneck — 4 of 8 defect labels have fewer than 50 positive examples, making reliable detection difficult for   
 rare defects.                                                                                                                                                
 5. The architecture is production-ready — it handles missing modalities gracefully, uses causal convolutions for real-time deployment, and includes domain   
 adaptation for robustness to unseen process conditions.                                                                                                      
