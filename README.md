This is the implementation of a novel knowledge tracing framework SYNSAINT.

Use one NVIDIA RTX A6000 GPU with 48G as memory, Python 3.9 and Pytorch 2.0.0 with cuda 11.7 for reproduction.

First, set up the environment.

```
pip install -r requirements_install.txt
```

Second, for dataset EdNet, download the original dataset from [https://github.com/riiid/ednet].

Run the following codes.

```
cd EdNet
```

```
python GNN_newSkillcluster_EdNet.py
```

```
python Hardness_EdNet.py
```

```
python train_sh.py
```
Third, for dataset Assistment2017, download the original dataset from [https://sites.google.com/view/assistmentsdatamining/dataset].

Run the following codes.

```
cd assist17
```

```
python GNN_newSkillcluster_Assist17.py
```

```
python Hardness_Assist17.py
```

```
python train_3.py
```
