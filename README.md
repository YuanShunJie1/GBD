# GBD
General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning

## Usage
"icdm" -> **TECB** backdoor attack.

"tifs" -> **BASL** backdoor attack.

We have implemented our backdoor detection method, GBD, under both attacks. Taking the detection of the BASL attack as an example, in the "tifs" directory, main.py is used to run the BASL backdoor attack, while test.py is used to perform backdoor detection using our approach. The ubd.py file in the "attackers" directory represents our method. First, you need to run **main.py** to launch the backdoor attack on VFL, and then run **test.py** to perform backdoor detection during the inference phase.

## Acknowledge
The code implemented in this work is based on some open-source projects related to VFL on GitHub. Thank you!

- **TECB:** He, Ying, et al. "Backdoor Attack Against Split Neural Network-Based Vertical Federated Learning." IEEE Transactions on Information Forensics and Security (2023).
- **BASL:** Chen, Peng, et al. "A practical clean-label backdoor attack with limited information in vertical federated learning." 2023 IEEE International Conference on Data Mining (ICDM). IEEE, 2023.
