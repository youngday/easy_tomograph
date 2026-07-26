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
![group_fbp](img_out/group_fbp.png)
![group_ir](img_out/group_ir.png)
fbp vs ir 
### astra_hybrid.py

fbp first and then to ir 
![astra_hybrid](img_out/astra_hybrid.png)
### tigre_plus_ir.py

tigre_hybrid
![tigre_hybrid](img_out/tigre_hybrid.png)

## install

note:tigre and tomophantom need install by github path
[install_by_uv](doc/install_by_uv.md)

## option gpu moniter

https://github.com/msminhas93/nviwatch
cargo install nviwatch

nviwatch
