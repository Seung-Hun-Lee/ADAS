## Pretrained Model

|    Target   | Cityscapes |  IDD | Mapillary | Model |
|:-----------:|:----------:|:----:|:---------:|:-----:|
| Source only |    37.2    | 36.1 |    37.9   |   *   |
| Warm up (1) |    44.3    | 40.5 |    41.9   |   *   |
| Warm up (2) |    46.3    | 43.9 |    47.6   |   *   |
|     BARS    |    53.5    | 49.8 |    52.8   |   *   |



## Training


### 1. Train source-only model

```
sh scripts/source_only_19.sh
```


### 2. Train MTDT-Net

```
sh scripts/train_MTDTNet_19.sh
```


### 3. Warm up (1) with MTDT-Net

```
sh scripts/da_MTDT_19.sh
```

### 4. Warm up (2) with AdaptSeg

```
sh scripts/da_AdaptSeg_MTDT_19.sh
```


### 5. Domain adaptation with BARS

```
sh scripts/da_BARS_19.sh
```


## Test
```
sh scripts/test_19.sh
```
