# <div align="center">SCIATICA</div>

> Repository for End term submission for Information Retrieval course (CS60092) offered in Spring semester 2023, Department of CSE, IIT Kharagpur.

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <!-- <img width="200" src="https://user-images.githubusercontent.com/86282911/230894496-b9402384-bf0a-4bf7-afbf-2207aa2d31be.png">
   -->
  <p align="center">
    <i>Research for research papers</i>
    <br />
    <br />
    <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />
    <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/>
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white">
    <br />
    <br />
    <a href="https://github.com/outer-rim/Sciatica/issues">Report Bug</a>
    ·
    <a href="https://github.com/outer-rim/Sciatica/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#directory-structure">Directory structure</a></li>
        <!-- <ul>
          <li><a href="#chromium-based-browsers">Chromium Based Browsers</a></li>
          <li><a href="#firefox">Firefox</a></li>
        </ul> -->
      </ul>
    </li>
    <li><a href="#colab-notebooks">Colab Notebooks</a></li>
    <!-- <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#miscelleneous">Miscelleneous</a></li>     -->
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project is an attempt of implementing and improving on the work of Sheshera Mysore, Tim O'Gorman, Andrew McCallum, Hamed Zamani titled `CSFCube - A Test Collection of Computer Science Papers for Faceted Query by Example`

The dataset can be found [here](https://github.com/iesl/CSFCube)

The paper describing the dataset can be accessed [here](https://arxiv.org/abs/2103.12906)

Demo video:

Team members:

- Ashwani Kumar Kamal - 20CS10011
- Hardik Pravin Soni - 20CS30023
- Shiladitya De - 20CS30061
- Sourabh Soumyakanta Das - 20CS30051

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

A quick introduction of the minimal setup you need to get the application up

```shell
pip install -r requirements.txt
streamlit run deploy.py
```

### Directory Structure

- Any `.ipynb` files that need to be run must be placed in this root directory which will contain the `/data` directory and `/Results` directory.

- The `data` directory contains the CSFCube dataset

```shell
.
├── abstracts-csfcube-preds.json
├── abstracts-csfcube-preds.jsonl
├── abstracts-csfcube-preds-no-unicode.jsonl
├── evaluation_splits.json
├── test-pid2anns-csfcube-background.json
├── test-pid2anns-csfcube-method.json
├── test-pid2anns-csfcube-result.json
└── test-pid2pool-csfcube.json
```

- The `Results` directory contains the embeddings generated from the models used

```shell
.
├── alberta
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-alberta-background-ranked.json
│   ├── test-pid2pool-csfcube-alberta-method-ranked.json
│   └── test-pid2pool-csfcube-alberta-result-ranked.json
├── allenai_specter
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-allenai_specter-background-ranked.json
│   ├── test-pid2pool-csfcube-allenai_specter-method-ranked.json
│   └── test-pid2pool-csfcube-allenai_specter-result-ranked.json
├── all_mpnet_base_v2
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-all_mpnet_base_v2-background-ranked.json
│   ├── test-pid2pool-csfcube-all_mpnet_base_v2-method-ranked.json
│   └── test-pid2pool-csfcube-all_mpnet_base_v2-result-ranked.json
├── bert_nli
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-bert_nli-background-ranked.json
│   ├── test-pid2pool-csfcube-bert_nli-method-ranked.json
│   └── test-pid2pool-csfcube-bert_nli-result-ranked.json
├── bert_pp
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-bert_pp-background-ranked.json
│   ├── test-pid2pool-csfcube-bert_pp-method-ranked.json
│   └── test-pid2pool-csfcube-bert_pp-result-ranked.json
├── distilbert_nli
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   ├── result.json
│   ├── test-pid2pool-csfcube-distilbert_nli-background-ranked.json
│   ├── test-pid2pool-csfcube-distilbert_nli-method-ranked.json
│   └── test-pid2pool-csfcube-distilbert_nli-result-ranked.json
└── ensemble
    ├── test-pid2pool-csfcube-ensemble-background-ranked.json
    ├── test-pid2pool-csfcube-ensemble-method-ranked.json
    └── test-pid2pool-csfcube-ensemble-result-ranked.json
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Colab Notebooks

- [Base Model](https://colab.research.google.com/drive/1PdvyNnlwA4eUyS6pTumNFI6OTbFY-GYc?usp=sharing)

This notebook contains the code for generating embeddings from the base models. Avoid running it as it takes a long time to run. The embeddings are already provided in the Googe Drive of IR Submission Files.

- [Fine Tuning DistilBERT (Grid Search)](https://colab.research.google.com/drive/1uax2mYhE2dTxSsZAx1jPaiy7TkTHNP7P?usp=sharing)

This is for the fine tuning of the Distilbert model. The results are already present in it. Avoid ruuning it as it takes a long time.

- [Ensembling models](https://colab.research.google.com/drive/1rOpEhRB_eAXny721EhDHkQ4l9kOkhhZ6?usp=sharing)

Run each cell of this jupyter notebook and at the second last cell change the queries as per choice and then run both the cells (itself and after it) and it gives the results.

- [IR Submission Files (Google Drive)](https://drive.google.com/drive/folders/1Wz8Pjxzp_hn4axTpviRWurgP6N57BzcK?usp=sharing)


Apart rom all this We are also submitting a zip of the local copies and reports of the .ipynb files which can be run locally. 
[Note] Please change the file directories strings in the notebooks appropriately to avoid any errors.

<p align="right">(<a href="#top">back to top</a>)</p>
