# LDMGI Image Clustering - Python Implementation

Reproduction of:  
**"Image Clustering Using Local Discriminant Models and Global Integration"**  
Yang et al., IEEE TIP 2010  
[Paper](https://ieeexplore.ieee.org/document/5454426)

## Features
- Implements LDMGI and K-Means clustering
- Supports 11 image datasets
- Evaluates with ACC/NMI metrics

## Project Structure
```
├── main.py              # Main script
├── ldmgi.py             # LDMGI implementation
├── kmeans.py            # K-Means implementation
├── data_loaders.py      # Dataset loaders function
├── base_loader.py       # Dataset loaders
└── data_samples.py      # Transform processed data to images
```

## Usage

### Supported Datasets

Datasets can be referenced by name or number:

| Number | Dataset Name    | 
|--------|-----------------|
| 1      | coil20          |
| 2      | jaffe           |
| 3      | pointing04      | 
| 4      | umist           | 
| 5      | yaleb           | 
| 6      | usps            | 
| 7      | mnist_t         | 
| 8      | mnist_s         | 
| 9      | mpeg7           | 
| 10     | umist_gabor     | 
| 11     | mpeg7_gray      | 

## Usage Examples

### Data Loading

```bash
# Load all datasets
python base_loader.py

# Load specific dataset
python base_loader.py [dataset_name]
```

### Sample Visualization

```sh
python data_samples.py [dataset_name]
```
### Clustering
```sh
# Run specific model on dataset (use name or number)
python main.py --dataset [name|number] --model [ldmgi|kmeans|1|2]

# Examples:
python main.py --dataset coil20 --model ldmgi    # by name
python main.py --dataset 1 --model 1             # by number (COIL20 + LDMGI)
python main.py --dataset all --model all         # run all combinations
```

### Output
```
├── ./data/processed/
├── ./image_samples/
├── ./results/
```



