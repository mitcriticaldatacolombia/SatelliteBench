# Multimodal Satellite Images Fusion for Dengue Prediction and Socioeconomic Analysis

- 📄 **Paper**: ([Online](https://link.springer.com/chapter/10.1007/978-3-031-82346-6_1))
- 🤗 **Dataset on HuggingFace**: ([MIT Critical data Colombia](https://huggingface.co/MITCriticalData))
## Overview
This repository contains the implementation of a data fusion framework that combines satellite images and tabular data for dengue prediction and socioeconomic analysis. The framework leverages variational autoencoders (VAE) for generating embeddings and employs an LSTM-based data fusion model to integrate time series of embeddings and metadata. The experiments in this repository demonstrate the effectiveness of the proposed approach in predicting dengue outbreaks and analyzing socioeconomic indicators like poverty and access to water.


## Table of Contents
- [Introduction](#introduction)
- [Satellite Extractor](#satellite_extractor)
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

![Fusion_Model_SatelliteBench-Final_Model](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/a5e6828a-dcca-4312-a0a1-dc71c9ce99b8)

## Dataset
The first step to use the code is to obtain the dataset that is used in each of the notebooks. You can find a demo dataset in `Demo_dataset/`. The demo dataset contains the files with the following data: satellite image embeddings generated using a variational autoencoder method with a Resnet 50 V2 backbone, meetadata and dengue cases, temperature, and precipitation. The satellite image embeddings is available for all the 12 bands, and RGB bands. The full dataset used that contains weekly satellite images of 81 cities in Colombia between 2016 and 2018, can be found in [Hugging Face](https://huggingface.co/MITCriticalData).

If you want to generate your own dataset of satellite images extracted for specific coordinates, you can use the satellite extractor framework. The framework fro satellite images extraction can be found in the following link: [satellite extractor](https://github.com/mitcriticaldatacolombia/satellite.extractor/tree/main).

## Installation
To use the code in this repository, follow these steps:

1. Clone the repository:

```
git clone https://github.com/mitcriticaldatacolombia/SatelliteBench.git
cd SatelliteBench
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


1. **Download your customized satellite images**: Customize satellite images to your specifications by selecting desired timestamps and Regions of Interest (ROIs). See notebook [here](https://github.com/mitcriticaldatacolombia/SatelliteBench/blob/main/download_images.ipynb)
1. **Embedding Generation:** Our project encompasses the sophisticated implementation of variational autoencoders (VAE), meticulously crafted to generate embeddings extracted from satellite images. Through the seamless integration of cutting-edge VAE technology, we ensure the precise extraction and representation of intricate spatial features, facilitating advanced analytics and insights.
1. **Data Fusion Model:** Within our framework, we introduce a robust LSTM-based data fusion model meticulously designed to amalgamate the temporal evolution of embeddings with comprehensive temporal metadata. This innovative approach empowers us to harness the power of longitudinal data streams, enabling a deeper understanding of complex spatial-temporal dynamics and facilitating enhanced predictive modeling capabilities.
1. **Dengue Prediction:** Leveraging the integrated embeddings derived from our data fusion model, we embark on a pioneering endeavor to forecast dengue outbreaks with unparalleled accuracy. By harnessing the multimodal nature of our data, we equip stakeholders with invaluable predictive insights, empowering proactive measures and interventions to mitigate the impact of dengue outbreaks and safeguard public health.
1. **Socioeconomic Analysis:** Our comprehensive framework extends beyond predictive analytics to encompass a multifaceted exploration of socioeconomic dynamics. Through the utilization of fused embeddings, we embark on an insightful journey to analyze critical socioeconomic indicators such as poverty levels and access to water resources. This holistic approach not only facilitates a nuanced understanding of societal challenges but also informs targeted interventions and policy decisions aimed at fostering sustainable development and societal well-being.

## Results
The results of the experiments are provided in the `results` directory. We include evaluation metrics and visualizations that demonstrate the performance of the data fusion framework for dengue prediction and socioeconomic analysis.

Geo-heatmap of Dengue Outbreak - Real 2019 Dec 23-29
![stA](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/bf8488d4-b34c-4e75-921b-f17164ecbbd4)
Geo-heatmap of Dengue Outbreak - Predicted 2019 Dec 23-29
![stB](https://github.com/mitcriticaldatacolombia/MIT_Multimodal_Satellite_Images_Fusion/assets/36363910/ffe1ee02-0a0c-43ef-a525-34356e4d9298)

## Colab Demo
We have also included a demo here [demo](https://colab.research.google.com/drive/1s28QdNin6lPOBPD6ibATNR2SZZQCjEY2?usp=sharing) on google colab. Runnable with free tier!

## Contributing

Your contributions to this repository are highly encouraged! Should you encounter any issues or have suggestions for improvements, we invite you to initiate an issue or submit a pull request. Your input is valued and greatly appreciated.

## License
This project is licensed under the [MIT License](LICENSE).
