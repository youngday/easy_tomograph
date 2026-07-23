# start

## 


# 激活环境
conda activate tomo-test

# 运行测试
python fbp_test.py



## dicom view 



```sh
conda activate tomo-test

# 列出目录下所有 DICOM 文件
python batch_view.py dicom_output/

# 显示元数据摘要表格
python batch_view.py dicom_output/ --info

# 逐张显示图像
python batch_view.py dicom_output/ --show

# 批量导出为 PNG
python batch_view.py dicom_output/ --save

# 查看单个文件
python batch_view.py --file dicom_output/recon_fbp.dcm


## tomopy

~/miniconda3/bin/conda install -n tomo-test -c conda-forge tomopy=1.15.3 -y 2>&1 | tail -15

## tomopy gpu 
# ASTRA + CUDA
conda install -c astra-toolbox astra-toolbox

# 验证 GPU 可用
python -c "import tomopy; print(tomopy.astra)"
