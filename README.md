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
![group_fbp](img_out/group_fbp.png)
![group_ir](img_out/group_ir.png)
fbp vs ir 
### fbp_plus_ir.py

fbp first and then to ir 
![fbp_plus_ir](img_out/fbp_plus_ir.png)
### tigre_plus_ir.py

tigre_plus_ir
![tigre_fbp_plus_ir](img_out/tigre_fbp_plus_ir.png)

## install

note:tigre and tomophantom need install by github path
[install_by_uv](install_by_uv.md)

## option gpu moniter

https://github.com/msminhas93/nviwatch
cargo install nviwatch

nviwatch
