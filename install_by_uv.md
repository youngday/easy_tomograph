# uv

## uv env

uv venv
uv init 
uv sync



## run

uv run fbp_vs_ir.py

## TIGRE

# 下载源码
wget https://github.com/CERN/TIGRE/archive/refs/tags/v3.1.3.tar.gz
tar xzf v3.1.3.tar.gz
cd TIGRE-3.1.3

## proxy
curl -x http://127.0.0.1:31181 -sL "https://github.com/CERN/TIGRE/archive/refs/tags/v3.1.3.tar.gz" -o /tmp/TIGRE-3.1.3.tar.gz && echo "Download OK" || echo "Download failed"

tar xzf /tmp/TIGRE-3.1.3.tar.gz -C /tmp && echo "Extract OK" && ls /tmp/TIGRE-3.1.3/
uv pip install numpy scipy matplotlib h5py tqdm 2>&1 | tail -5
uv pip install /tmp/TIGRE-3.1.3 2>&1


## tomophantom

HTTPS_PROXY=http://127.0.0.1:31181 uv add git+https://github.com/dkazanc/TomoPhantom.git 2>&1 | tail -10


内置体模（2D 库）

| 编号 | 名称 | 说明 |
|:---:|:----|:------|
| 01 | Classical Shepp-Logan | 经典 10 椭圆体模 |
| 02 | Piecewise-Smooth S-L | 平滑/分段组合版 |
| 03 | Defrise | 竖直椭圆条测试体模 |
| **04** | **QRM** | **多椭圆测试体模（我们在用）** |
| 05-15 | 各种组合体模 | 高斯/锥体/矩形等

# 编译安装
uv pip install numpy scipy matplotlib h5py tqdm
uv pip install .
