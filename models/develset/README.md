# Deep levelset OPC


The levelset_net is the DSO part of the DevelSet paper.

## Dependency


0. pytorch

```
conda install pytorch torchvision torchaudio -c pytorch
```

1. Pytorch-lightning

```
pip install pytorch-lightning
```



2. hydra

```
pip install hydra-core --upgrade
```

3. cupy
see [cupy.dev](https://cupy.dev/)

for me:

```
pip install cupy-cuda100
```

4. TensorBoardX for visualization

```
pip install tensorboardx
```

## How to run.

1. Create a conda environment and install `pytorch-lightning`, `hydra`, `cupy`.
2. The config file is in the config/main.yaml.
   1. Here please add your own dataset config file.
   2. Such as hongduo_iccad13_data.yaml
   3. copy the iccad13_data.yaml, change the basic path to your own path
3. Run `python train.py`
4. The outputs are in the `outputs` folder

Enjoy!

## Architecture:

All the code are under the `levelset_net/src`
### Lithography model:

`levelset_net/src/models/litho_layer.py`

This is the CUDA version of the litho model in Mosaic paper.


### Mosaic solver:

`levelset_net/src/models/mosaic.py`

This is the mosaic solver in Mosaic paper.


### DSO solver:

`levelset_net/src/models/develset.py`

This is the develset sovelr in DevelSet paper.