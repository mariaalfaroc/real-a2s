<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Insights into end-to-end audio-to-score transcription with real recordings: A case study with saxophone works</h1>

<h4 align="center">Full text available <a href="https://www.isca-speech.org/archive/interspeech_2023/martinezsevilla23_interspeech.html" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

End-to-end Audio-to-Score (A2S) transcription aims to retrieve a score that encodes the music content of an audio recording in a single step. Due to the recentness of this formulation, the existing works have exclusively addressed controlled scenarios with synthetic data that fail to provide conclusions applicable to real-world cases. In response to this gap in the literature, **we introduce a compilation of recorded saxophone performances together with their digital music scores and pose several experimental scenarios involving real and synthetic data**.


> More in-depth information about this dataset can be found [here](https://grfia.dlsi.ua.es/audio-to-score/) or in our [research paper](https://grfia.dlsi.ua.es/audio-to-score/).


The obtained results confirm the adequacy of this A2S framework to deal with real data as well as proving the relevance of leveraging synthetic interpretations to improve the recognition rate in scenarios with real-data scarcity.

## How To Use

To run the code, you'll need to meet certain requirements which are specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred.

Once you have prepared your environment (either a Docker container or a virtual environment), you are ready to begin. Execute the [`run_experiments.sh`](run_experiments.sh) script to replicate the experiments from our work:

```bash
$ sh run_experiments.sh
```

## Citations

```bibtex
@inproceedings{martinez2023insights,
  title     = {{Insights into end-to-end audio-to-score transcription with real recordings: A case study with saxophone works}},
  author    = {Mart{\'\i}nez-Sevilla, Juan C. and Alfaro-Contreras, Mar{\'\i}a and Valero-Mas, Jose J. and Calvo-Zaragoza, Jorge},
  booktitle = {{Proceedings of the 24th INTERSPEECH Conference}},
  year      = 2023,
  pages     = {2793--2797},
  month     = aug,
  address   = {Dublin, Ireland},
  doi       = {10.21437/Interspeech.2023-88}
}
```

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License
This work is under a [MIT](LICENSE) license.