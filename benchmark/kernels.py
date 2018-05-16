import torch
import support.kernel as kernel_factory
import numpy as np
import matplotlib.pyplot as plt


class BenchRunner:
    def __init__(self, kernel, tensor_size, tensor_initial_device='cpu'):
        # tensor_size = (4, 3)
        # print('BenchRunner::__init()__ getting kernel and initializing tensors with size ' + str(tensor_size))
        self.kernel_instance = kernel_factory.factory(kernel, kernel_width=1.)

        self.x = torch.rand(tensor_size, device=torch.device(tensor_initial_device))
        self.y = self.x.clone()
        self.p = torch.ones(tensor_size, device=torch.device(tensor_initial_device))

        # run once for warm-up: cuda pre-compile
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)
        # print('BenchRunner::__init()__ done')

    def run(self):
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)

        # print(self.res)
        # move to CPU
        # self.res.to(torch.device('cpu'))

        # self.res = None
        # torch.cuda.empty_cache()

    def __exit__(self):
        print('BenchRunner::__exit()__')


def build_setup():
    # kernels = ['CudaExactTorch']
    kernels = ['exact']
    # initial_devices = ['cpu', 'cuda:0']
    initial_devices = ['cpu']
    tensor_sizes = [(4, 3), (16, 3), (32, 3), (64, 3), (128, 3), (256, 3), (512, 3)]
    # tensor_sizes = [(4, 3), (16, 3), (32, 3)]
    setups = []

    for k, d, t in [(k, d, t) for k in kernels for d in initial_devices for t in tensor_sizes]:
        bench_setup = '''
from __main__ import BenchRunner
bench = BenchRunner('{kernel}', {tensor}, '{device}')
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
        res['data'] = timeit.repeat("bench.run()", number=1000, repeat=3, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])

        print(res)
        results.append(res)

    # print('cpu: ' + str(timeit.repeat("bench.run()", number=50000, repeat=3, setup=setup_cpu)))
    # print('cuda: ' + str(timeit.repeat("bench.run()", number=50000, repeat=3, setup=setup_cuda)))

    # cpu_res = [r['max'] for r in results if r['setup']['device'] == 'cpu']
    # cuda_res = [r['max'] for r in results if r['setup']['device'] == 'cuda:0']
    # assert(len(cpu_res) == len(cuda_res))

    fig, ax = plt.subplots()
    ax.set_yscale('log', nonposy='clip')

    for d in set([r['setup']['device'] for r in results]):
        res_data = [r['max'] for r in results if r['setup']['device'] == d]
        index = np.arange(len(res_data))
        bar_width = 0.35
        opacity = 0.4
        ax.bar(index, res_data, bar_width, alpha=opacity, label=d)

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=opacity, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=opacity, color='g', label='cuda')

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime by device/size')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([r['setup']['tensor_size'] for r in results if r['setup']['device'] == 'cpu'])
    ax.legend()

    fig.tight_layout()

    plt.show()
