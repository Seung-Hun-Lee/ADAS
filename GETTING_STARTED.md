## Pretrained Model

|    Target   | Cityscapes |  IDD | Mapillary | Model |
|:-----------:|:----------:|:----:|:---------:|:-----:|
| Source only |    37.2    | 36.1 |    37.9   |   [model](https://drive.google.com/file/d/1Ng7SZ16PywNoG_PiudnvrYluZTrPIHj_/view?usp=drive_link)   |
| Warm up (1) |    44.3    | 40.5 |    41.9   |   [model](https://drive.google.com/file/d/1CpTzScu4N3ofW9NHnFEf6l7xT9bBIj0E/view?usp=drive_link)   |
| Warm up (2) |    46.3    | 43.9 |    47.6   |   [model](https://drive.google.com/file/d/1_ebV2r8qBX21zuh52QzpMvpgVsoHoa5H/view?usp=drive_link)   |
|     BARS    |    53.5    | 49.8 |    52.8   |   [model](https://drive.google.com/file/d/1ujP3oSACTp-nwGs_Kcway1pF07_9dv2k/view?usp=drive_link)   |



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
