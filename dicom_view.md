# start

## 


# 激活环境
conda activate tomo-test

# 运行测试
python fbp_test.py



## dicom view 

```sh

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
