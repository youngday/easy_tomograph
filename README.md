# easy learn tomographic reconstruction 
* 数学基础 Radon 变换 + 傅里叶切片定理
* fbp vs ir
* fbp add ir
* based gpu (astra-toolbox and tigre)
* tomophantom (体模生成库)
* rtk (not run on gpu directly,now not used.)
* axial and helical (now ,just axial)
## env 

uv cuda cupy astra tigre tomophantom matplotlib

## examples


### batch_view.py
 dicom to picture 
### fbp_vs_ir.py 

* fbp filters
<!--![group_fbp](img_out/group_fbp.png)-->
<!--![group_ir](img_out/group_ir.png)-->
fbp vs ir 
## 2d fan hybrid
sirt os-sart
### astra_hybrid.py
![astra_hybrid](img_out/astra_hybrid.png)
### tigre_hybrid.py
![tigre_hybrid](img_out/tigre_hybrid.png)

## 3d cone hybrid
sirt os-sart
### astra_cone_hybrid.py
![astra_cone_hybrid](img_3d_axial/astra_cone_hybrid.png)
### tigre_cone_hybrid.py
tigre_hybrid
![tigre_cone_hybrid](img_3d_axial/tigre_cone_hybrid.png)

## install

note:tigre and tomophantom need install by github path
[install_by_uv](doc/install_by_uv.md)

## option gpu moniter

https://github.com/msminhas93/nviwatch
cargo install nviwatch

nviwatch
