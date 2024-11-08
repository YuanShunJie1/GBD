# GBD
General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning

## Usage
The "icdm" and "tifs" folders correspond to the **TECB** backdoor attack and the **BASL** backdoor attack, respectively. We have implemented our backdoor detection method, GBD, under both attacks. Taking the detection of the BASL attack as an example, in the "tifs" directory, main.py is used to run the BASL backdoor attack, while test.py is used to perform backdoor detection using our approach. The ubd.py file in the "attackers" directory represents our method. First, you need to run **main.py** to launch the backdoor attack on VFL, and then run **test.py** to perform backdoor detection during the inference phase.

## 用法
icdm和tifs这两个文件夹分别对应着**TECB**后门攻击和**BASL**后门攻击。我们分别在这两个攻击下实现了我们后门检测方法GBD。我们以检测BASL攻击为例子。在tifs目录中，main.py代表运行BASL后门攻击，test.py代表利用我们的方案进行后门检测，attackers目录下的ubd.py文件表示我们方案。首先你需要运行**main.py**对VFL发起后门攻击，其次运行**test.py**就可以在推理阶段实现后门检测。

## Thanks
The implementation of the code in this work is based on some open-source projects related to VFL on GitHub. Thank you!
