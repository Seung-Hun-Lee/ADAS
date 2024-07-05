## Train


1. Source only
```
sh scripts
```


1. MTDT-Net (image to multi-target images)
```
python MTDT_train.py -D [datasets] -N [num_classes] --iter [itrations] --ex [experiment_name]
example) python MTDT_train.py -D G C I M -N 19 --iter 3000 --ex MTDT_19
```
You can see the translated images on tensorboard.
```
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```
2. Pretraining (option)
```
python Source_Only.py -D G C I M -N 19 --iter 100000 --ex Source_Only_19
```
3. Domain Adaptatoin with MTDT-Net
```
python MTDT_DA.py -D G C I M -N 19 --iter 200000 --load_mtdt checkpoint/MTDT_19/3000/netMTDT.pth --ex MTDT_DA_19
(If you have pretrained model, you can add '--load_seg checkpoint/Source_Only_19/100000/netT.pth' option.)
```
4. Domain Adaptation with BARS
```
python BARS_DA.py -D G C I M -N 19 --iter 200000 --load_mtdt checkpoint/MTDT_19/3000/netMTDT.pth --load_seg checkpoint/MTDT_DA_19/200000/netT.pth --ex BARS_DA_19
```

## Test
```
python test.py -D G C I M -N 19 --load_seg [trained network]
```
