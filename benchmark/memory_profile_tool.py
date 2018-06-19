import ctypes
import GPUtil
import time
from threading import Thread
import resource
import sys


# _cudart = ctypes.CDLL('libcudart.so')
#
#
# def start_cuda_profile():
#     # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
#     # the return value will unconditionally be 0. This check is just in case it changes in
#     # the future.
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception("cudaProfilerStart() returned %d" % ret)
#
#
# def stop_cuda_profile():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception("cudaProfilerStop() returned %d" % ret)


class MemoryProfiler(Thread):
    def __init__(self, freq=1.):
        Thread.__init__(self)
        self.freq = freq
        self.run_flag = True
        self.data = []

    def run(self):
        # print('MemoryProfiler::run()')
        while self.run_flag:
            # print('MemoryProfiler::append()')
            self.data.append({
                'ram': self.ram_usage(),
                'gpu_ram': GPUtil.showUtilization(all=True)
            })
            time.sleep(self.freq)

    def stop(self):
        # print('MemoryProfiler::stop()')
        self.run_flag = False
        self.join()
        return self.data

    def clear(self):
        self.data.clear()

    def ram_usage(self):
        # TODO: this need to be checked !
        # return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rusage_denom = 1024.
        if sys.platform == 'darwin':
            # ... it seems that in OSX the output is different units ...
            rusage_denom = rusage_denom * rusage_denom
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom


def start_memory_profile(freq=1.):
    ret = MemoryProfiler(freq)
    ret.start()
    return ret


def stop_memory_profile(memory_profiler):
    return list(memory_profiler.stop())


def stop_and_clear_memory_profile(memory_profiler):
    ret = list(memory_profiler.stop())
    clear_memory_profile(memory_profiler)
    return ret


def clear_memory_profile(memory_profiler):
    memory_profiler.clear()



