# Useful commands

##### Drun for interactive command

```bash
drun -a SOCISP -i Semantic_ADAS  -c bash -q gpu_debug_q -I -v /opt/dockermounts/algo-datasets:/mnt/datasets -p 8123:8181 -p 6123:6006 --pull -R "select[ostype=UBUNTU16.04 && g_model=GeForceGTX1080Ti]"
```

this will open bash on Semantic_ADAS container, map port 8181 to 8123 and as for a certain OS and GPU. 

##### Drun for noninteractive

```bash
drun -a SOCISP -i Semantic_ADAS -c "su yotampe -c'path_to_bash-script.sh'" -q gpu_deep_train_high_q -v /opt/dockermounts/algo-datasets:/mnt/datasets --pull -R "select[ostype=UBUNTU16.04 && g_model=GeForceGTX1080Ti]"

```

##### Open pycharm

```bash
opt/pycharm-community-2018.3.4/bin/pycharm.sh&
```

##### Open Jupyter notebook

```bash
/etc/anaconda3/bin/jupyter notebook --ip=0.0.0.0 --port=8181
# Then go to the correct link in your browser e.g http://gpu35-dt:8123 (depending on the port you mapped to and your GPU number) and insert the token number.
```

##### A container with PyTorch and torchdiffeq

```bash
drun -a SOCISP -i Semantic_ADAS_torch_1.1.0_torchdiffeq -c bash -q gpu_debug_q -I -v /opt/dockermounts/algo-datasets:/mnt/datasets -p 8123:8181 -p 6123:6006 --pull -R "select[ostype=UBUNTU16.04 && g_model=GeForceGTX1080Ti]"
```

##### A Matlab container

```bash
drun -I -q gpu_debug_q -a SOCISP -i cuda8-cudnn6-ubu14-caffe-tf141-matlab-pycharm -c "su yotampe" --pull
```

##### Create a new container

```bash
dcommit -j YOUR_JOB_ID -n NEW_CONTAINER_NAME
```

##### Configure proxy server on container (bash)

```bash
export {http,https,ftp}_proxy=http://dlp2-wcg01:8080
export {HTTP,HTTPS}_PROXY=http://dlp2-wcg01:8080
```

##### Export proxy (for tensorboard)

```bash
export http_proxy=http://106.199.15.25:8080
export https_proxy=http://106.199.15.25:8080
```

