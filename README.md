# easy learn tomographic reconstruction 
* 数学基础 Radon 变换 + 傅里叶切片定理
* fbp vs ir
* fbp add ir
* based gpu (astra-toolbox and tigre)
* rtk (not run on gpu directly,now not used.)
* axial and helical (now ,just axial)
## env 
conda with cupy (uv not ok ,for cuda and other),todo:uv 
## examples


### batch_view.py
 dicom to picture 
### fbp_vs_ir.py 

* fbp filters
![alt text](img_out/fbp_group.png)
* ![alt text](img_out/gpu_ir_group.png)
fbp vs ir 
### fbp_plus_ir.py
fbp first and then to ir 
![alt text](img_out/fbp_plus_ir.png)

## install

![install_by_uv](install_by_uv.md)
