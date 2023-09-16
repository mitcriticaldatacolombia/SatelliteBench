# Multimodal Satellite Images Fusion for Dengue Prediction and Socioeconomic Analysis

## Overview
This repository contains the implementation of a data fusion framework that combines satellite images and tabular data for dengue prediction and socioeconomic analysis. The framework leverages variational autoencoders (VAE) for generating embeddings and employs an LSTM-based data fusion model to integrate time series of embeddings and metadata. The experiments in this repository demonstrate the effectiveness of the proposed approach in predicting dengue outbreaks and analyzing socioeconomic indicators like poverty and access to water.

For the data extraction also check the [satellite extractor](https://github.com/mitcriticaldatacolombia/satellite.extractor/blob/main/README.md)


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Colab Demo](https://colab.research.google.com/drive/1s28QdNin6lPOBPD6ibATNR2SZZQCjEY2?usp=sharing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
Data fusion of satellite images and tabular data has shown promising results in various domains. In this project, we propose a multimodal data fusion framework that combines the visual information from satellite images with contextual tabular data to predict dengue outbreaks and analyze socioeconomic indicators. The framework consists of two main components: (1) Variational Autoencoders (VAE) for generating embeddings from satellite images and (2) an LSTM-based data fusion model to integrate the embeddings with temporal metadata.

![Chenwei_Fusion_Model_SatelliteBench-Final_Model](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/a5e6828a-dcca-4312-a0a1-dc71c9ce99b8)


## Installation
To use the code in this repository, follow these steps:

1. Clone the repository:

```
git clone https://github.com/dsrestrepo/MIT_Multimodal_Satellite_Images_Fusion.git
cd MIT_Multimodal_Satellite_Images_Fusion
```

2. Set up the Python environment. It is recommended to use a virtual environment:

```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage
Once you have installed the required dependencies, you can use the code for generating embeddings, training the fusion model, and conducting experiments on dengue prediction and socioeconomic analysis.

## Experiments
This repository includes the following experiments:

1. **Embedding Generation:** Implementation of variational autoencoders (VAE) for generating embeddings from satellite images.
2. **Data Fusion Model:** Implementation of the LSTM-based data fusion model that integrates the time series of embeddings with temporal metadata.
3. **Dengue Prediction:** Using the fused embeddings to predict dengue outbreaks based on the multimodal data.
4. **Socioeconomic Analysis:** Utilizing the fused embeddings to analyze socioeconomic indicators like poverty and access to water.

## Results
The results of the experiments are provided in the `results` directory. We include evaluation metrics and visualizations that demonstrate the performance of the data fusion framework for dengue prediction and socioeconomic analysis.

Geo-heatmap of Dengue Outbreak - Real 2019 Dec 23-29
![stA](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/bf8488d4-b34c-4e75-921b-f17164ecbbd4)
Geo-heatmap of Dengue Outbreak - Predicted 2019 Dec 23-29
![stB](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/ffe1ee02-0a0c-43ef-a525-34356e4d9298)

## Colab Demo
We have also included a demo here [demo](https://colab.research.google.com/drive/1s28QdNin6lPOBPD6ibATNR2SZZQCjEY2?usp=sharing) on google colab. Runnable with free tier!

## Contributing
We welcome contributions to this repository! If you find any issues or have improvements to suggest, please open an issue or a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or inquiries, please feel free to reach out to David Restrepo at davidres@mit.edu.

