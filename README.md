# 📌 General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning

This repository contains the official implementation of the paper:

> **General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning**

In this project, we demonstrate a general framework for detecting backdoor attacks in vertical federated learning (VFL) based on split neural networks (SplitNN). Specifically, we take the **BASL** backdoor attack as a case study and implement both our proposed defense method **GBD** and a comparison baseline **VFLIP**.

---

## 🧪 Implemented Methods

- **GBD (General Backdoor Detection)**: Our proposed method, which identifies and mitigates test-time backdoors in SplitNN-based VFL settings.
- **VFLIP**: A comparative method taken from the paper _"VFLIP: A Backdoor Defense for Vertical Federated Learning via Identification and Purification"_.

---

## 🚀 How to Run

1. **Download the dataset**  
   Place the required datasets into the `dataset/` directory.

2. **Launch the BASL backdoor attack**  
   Run the following script to simulate the backdoor attack:
   ```bash
   python basl_attack.py

3. **Perform backdoor detection**  
   You can evaluate detection performance using either method:
   ```bash
   python detect_gbd.py
   python detect_vflip.py


## 📖 Reference Papers

- **BASL attack**  
  Y. He, Z. Shen, J. Hua, Q. Dong, J. Niu, W. Tong, X. Huang, C. Li, and S. Zhong, “Backdoor attack against split neural network-based vertical federated learning,” *IEEE Transactions on Information Forensics and Security*, vol. 19, pp. 748–763, Oct 2024.

- **VFLIP defense**  
  Y. Cho, W. Han, M. Yu, Y. Lee, H. Bae, and Y. Paek, “VFLIP: A backdoor defense for vertical federated learning via identification and purification,” in *Proceedings of the 29th European Symposium on Research in Computer Security (ESORICS 2024)*, Bydgoszcz, Poland, Sep 2024, pp. 291–312.

## 🛠️ Environment Requirements
- **Python ≥ 3.7**
- **PyTorch ≥ 1.8**


## 🙏 Acknowledgements

This repository partially builds upon the open-source projects: https://github.com/WWlnZSBMaXU/Triggerless, https://github.com/13thDayOfLunarMay/TECB-attack, https://github.com/Mr-Ace-1997/Backdoor-Attack-Against-Split-Neural-Network-Based-Vertical-Federated-Learning

We sincerely thank the original authors for making their code publicly available. The original project is licensed under the MIT License, and we have retained the original license in this repository in accordance with its terms. Based on their implementation, we have made modifications and extensions to support our proposed method and experimental settings.

