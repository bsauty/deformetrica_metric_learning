import torch
import support.kernel as kernel_factory
import numpy as np
import matplotlib.pyplot as plt


class CudaExactTorch:
    def __init__(self, tensor_size, tensor_initial_device='cpu'):
        # tensor_size = (4, 3)
        # print('CudaExactTorch::__init()__ getting kernel and initializing tensors with size ' + str(tensor_size))
        self.kernel_instance = kernel_factory.factory(kernel_factory.Type.CUDA_EXACT_TORCH, kernel_width=1.)

        self.x = torch.rand(tensor_size, device=torch.device(tensor_initial_device))
        self.y = self.x.clone()
        self.p = torch.ones(tensor_size, device=torch.device(tensor_initial_device))

        # run once for warm-up: cuda pre-compile
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)
        # print('CudaExactTorch::__init()__ done')

    def run(self):
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)

        # print(self.res)
        # move to CPU
        # self.res.to(torch.device('cpu'))

        # self.res = None
        # torch.cuda.empty_cache()

    def __exit__(self):
        print('CudaExactTorch::__exit()__')


def build_setup():
    kernels = ['CudaExactTorch']
    initial_devices = ['cpu', 'cuda:0']
    tensor_sizes = [(4, 3), (16, 3), (32, 3), (64, 3), (128, 3), (256, 3), (512, 3)]
    # tensor_sizes = [(4, 3), (16, 3), (32, 3)]
    setups = []

    for k, d, t in [(k, d, t) for k in kernels for d in initial_devices for t in tensor_sizes]:
        bench_setup = '''
from __main__ import CudaExactTorch
instance = {kernel}({tensor}, '{device}')
            '''.format(kernel=k, tensor=str(t), device=d)

        setups.append({'kernel': k, 'device': d, 'tensor_size': t, 'bench_setup': bench_setup})
    return setups


if __name__ == "__main__":
    import timeit

    results = []

    for setup in build_setup():
        print('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        res['data'] = timeit.repeat("instance.run()", number=500, repeat=3, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])

        print(res)
        results.append(res)

    # print('cpu: ' + str(timeit.repeat("instance.run()", number=50000, repeat=3, setup=setup_cpu)))
    # print('cuda: ' + str(timeit.repeat("instance.run()", number=50000, repeat=3, setup=setup_cuda)))

    # plt.interactive(False)
    # plt.plot([r['setup']['device'] for r in results], [r['max'] for r in results])

    cpu_res = [r['max'] for r in results if r['setup']['device'] == 'cpu']
    cuda_res = [r['max'] for r in results if r['setup']['device'] == 'cuda:0']
    assert(len(cpu_res) == len(cuda_res))

    fig, ax = plt.subplots()

    index = np.arange(len(cpu_res))
    bar_width = 0.35
    opacity = 0.4

    bar1 = ax.bar(index, cpu_res, bar_width, alpha=opacity, color='b', label='cpu')
    bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=opacity, color='g', label='cuda')

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Runtime')
    ax.set_title('Runtime by device/size')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([r['setup']['tensor_size'] for r in results if r['setup']['device'] == 'cpu'])
    ax.legend()

    fig.tight_layout()

    plt.show()
