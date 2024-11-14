<!-- This template is adapated from https://github.com/othneildrew/Best-README-Template -->

[![Issues][issues-shield]][issues-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SoSci-SimLearn</h3>

  <p align="center">
    Train sentence-embeddings using contrastive learning.
    <br />
    <br />
    <a href="https://github.com/e-tornike/sosci-simlearn/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/e-tornike/sosci-simlearn/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository allows you to train custom sentence embeddings using [quaterion](https://github.com/qdrant/quaterion).

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Install [python](https://www.python.org/downloads/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/e-tornike/sosci-simlearn.git
   cd sosci-simlearn
   ```
2. Create a virtual environment
   ```
   poetry init
   ```
3. Install python packages
   ```
   poetry install
   ```

<!-- USAGE EXAMPLES -->
## Usage

Run the following bash script to start training:
```
bash run.sh
```

<!-- LICENSE -->
## License

MIT license. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Tornike Tsereteli - tsereteli.tornike@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/e-tornike/sosci-simlearn.svg?style=for-the-badge
[issues-url]: https://github.com/e-tornike/sosci-simlearn/issues
[license-shield]: https://img.shields.io/github/license/e-tornike/sosci-simlearn.svg?style=for-the-badge
[license-url]: https://github.com/e-tornike/sosci-simlearn/blob/master/LICENSE
[product-screenshot]: images/screenshot.png
