"""
TomoPhantom QRM 体模生成器
===========================
生成 Model 4 (QRM) 体模并映射到 CT 值范围
"""
import tomophantom, os, numpy as np
from tomophantom import TomoP2D

def make_phantom(N=512, model=4, hu_range=2000):
    """
    生成 TomoPhantom 体模并映射到 HU 值

    参数:
        N: 体模尺寸 (N×N)
        model: 体模编号 (1-15), 默认 4=QRM
        hu_range: HU 范围, 默认 2000 (-1000 ~ +1000)

    返回:
        ct: (N, N) float32, HU 值
        circ_mask: 圆形掩膜
    """
    lib = os.path.join(os.path.dirname(tomophantom.__file__),
                       'phantomlib', 'Phantom2DLibrary.dat')
    ph = TomoP2D.Model(model, N, lib)

    # 映射到 HU: 假设 ph ∈ [0, max_val]
    max_val = ph.max()
    ct = (ph - max_val / 2) * hu_range / max_val
    ct = ct.astype(np.float32)

    Y, X = np.ogrid[:N, :N]
    head_r = int(N * 0.46)
    circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= head_r ** 2
    ct[~circ_mask] = -1000

    return ct, circ_mask


if __name__ == '__main__':
    ct, mask = make_phantom()
    print(f'体模: {ct.shape}, 范围 [{ct.min():.0f}, {ct.max():.0f}]')

    import matplotlib.pyplot as plt
    plt.imshow(ct, cmap='gray', vmin=-200, vmax=600)
    plt.colorbar(label='HU')
    plt.title('QRM Phantom (Model 4)')
    plt.axis('off')
    plt.savefig('img_out/qrm_phantom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ img_out/qrm_phantom.png')
