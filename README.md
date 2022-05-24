# ACTOR



## Installation :construction_worker:
### 1. Create conda environment

```

conda create -n myactor python=3.8
conda activate myactor


pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


pip install tensorboard
pip install matplotlib
pip install ipdb
pip install sklearn
pip install pandas
pip install tqdm
pip install imageio
pip install pyyaml
pip install smplx
pip install chumpy

conda install -c fvcore -c iopath -c conda-forge fvcore iopath

conda install -c bottler nvidiacub

conda install pytorch3d -c pytorch3d

pip install timm==0.3.2
```




Running command

```
sbatch run_uestc_alpha05.sh
```


### 2. Download the datasets
**For all the datasets, be sure to read and follow their license agreements, and cite them accordingly.**

For more information about the datasets we use in this research, please check this [page](DATASETS.md), where we provide information on how we obtain/process the datasets and their citations. Please cite the original references for each of the datasets as indicated.

Please install gdown to download directly from Google Drive and then:
```bash
bash prepare/download_datasets.sh
```

**Update**: Unfortunately, the NTU13 dataset (derived from NTU) is no longer available.



## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
