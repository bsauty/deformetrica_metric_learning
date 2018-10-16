import os

import GPUtil
import torch
import torch.multiprocessing as mp
import numpy as np


from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject


def get_torch_dtype(t):
    """
    32-bit floating point	torch.float32 or torch.float	torch.FloatTensor	torch.cuda.FloatTensor
    64-bit floating point	torch.float64 or torch.double	torch.DoubleTensor	torch.cuda.DoubleTensor
    16-bit floating point	torch.float16 or torch.half	torch.HalfTensor	torch.cuda.HalfTensor
    8-bit integer (unsigned)	torch.uint8	torch.ByteTensor	torch.cuda.ByteTensor
    8-bit integer (signed)	torch.int8	torch.CharTensor	torch.cuda.CharTensor
    16-bit integer (signed)	torch.int16 or torch.short	torch.ShortTensor	torch.cuda.ShortTensor
    32-bit integer (signed)	torch.int32 or torch.int	torch.IntTensor	torch.cuda.IntTensor
    64-bit integer (signed)	torch.int64 or torch.long	torch.LongTensor	torch.cuda.LongTensor

    cf: https://pytorch.org/docs/stable/tensors.html
    :param t:
    :return:
    """
    if t in ['float32', np.float32]:
        return torch.float32
    if t in ['float64', np.float64]:
        return torch.float64
    if t in ['float16', np.float16]:
        return torch.float16
    if t in ['uint8', np.uint8]:
        return torch.uint8
    if t in ['int8', np.int8]:
        return torch.int8
    if t in ['int16', np.int16]:
        return torch.int16
    if t in ['int32', np.int32]:
        return torch.int32
    if t in ['int64', np.int64]:
        return torch.int64

    assert t in [torch.float32, torch.float64, torch.float16, torch.uint8,
                 torch.int8, torch.int16, torch.int32, torch.int64]
    return t


def move_data(data, device='cpu', dtype=None, requires_grad=False):
    """
    Move given data to target Torch Tensor to the given device
    :param data:    TODO
    :param device:  TODO
    :param requires_grad: TODO
    :param dtype:  dtype that the returned tensor should be formatted as.
                   If not defined, the same dtype as the input data will be used.
    :return: Torch.Tensor
    """
    assert data is not None, 'given input data cannot be None !'
    assert device is not None, 'given input device cannot be None !'

    if dtype is None:
        dtype = get_torch_dtype(data.dtype)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).type(dtype).to(device)
    if isinstance(data, list):
        data = torch.Tensor(data, dtype=dtype, device=device)

    assert isinstance(data, torch.Tensor), 'Expecting Torch.Tensor instance'

    # handle requires_grad flag
    if requires_grad:
        # user wants grad
        if data.requires_grad is False:
            data.requires_grad_()
        # else data already has requires_grad flag to True
    else:
        # user does not want grad
        if data.requires_grad is True:
            data.detach_()
        # else data already has requires_grad flag to False

    # move data to device. Note: tensor.to() does not move if data is already on target device
    return data.to(device)


def convert_deformable_object_to_torch(deformable_object, device='cpu'):
    # bounding_box
    assert isinstance(deformable_object, DeformableMultiObject)
    assert deformable_object.bounding_box is not None
    if not isinstance(deformable_object.bounding_box, torch.Tensor):
        deformable_object.bounding_box = move_data(deformable_object.bounding_box, device=device)
    deformable_object.bounding_box = deformable_object.bounding_box.to(device)

    # object_list
    for i, _ in enumerate(deformable_object.object_list):
        if not isinstance(deformable_object.object_list[i].bounding_box, torch.Tensor):
            deformable_object.object_list[i].bounding_box = move_data(deformable_object.object_list[i].bounding_box, device=device)
        deformable_object.object_list[i].bounding_box = deformable_object.object_list[i].bounding_box.to(device)

        if not isinstance(deformable_object.object_list[i].connectivity, torch.Tensor):
            deformable_object.object_list[i].connectivity = move_data(deformable_object.object_list[i].connectivity, device=device)
        deformable_object.object_list[i].connectivity = deformable_object.object_list[i].connectivity.to(device)

        if not isinstance(deformable_object.object_list[i].points, torch.Tensor):
            deformable_object.object_list[i].points = move_data(deformable_object.object_list[i].points, device=device)
        deformable_object.object_list[i].points = deformable_object.object_list[i].points.to(device)

    return deformable_object


def get_best_device(process_per_gpu=1):
    """

    :param process_per_gpu: Set the number of processes that are to use the same GPU.
                            This can be increased if the GPU can allocate sufficient memory.
    :return:    Best device. can be: 'cpu', 'cuda:0', 'cuda:1' ...
    """
    device = 'cpu'
    if torch.cuda.is_available():
        '''
        PoolWorker-1 will use cuda:0
        PoolWorker-2 will use cuda:1
        PoolWorker-3 will use cuda:2
        etc...
        '''
        # TODO: Use GPUtil to check if GPU memory is full
        # TODO: only use CPU if mp queue is still quite full (eg > 50%), else leave work for GPU

        process_id = int(mp.current_process().name.split('-')[1])   # eg: PoolWorker-0
        for device_id in range(torch.cuda.device_count()):
            # pool_worker_id = device_id
            if process_id < process_per_gpu:
                device = 'cuda:' + str(device_id)
                break
                # if mp.current_process().name == 'PoolWorker-' + str(pool_worker_id):
                #     device = 'cuda:' + str(device_id)
                #     break

        # try:
        #     # device_id = GPUtil.getFirstAvailable(order='first', maxMemory=0.5, attempts=1, verbose=False)[0]
        #     # device = 'cuda:' + str(device_id)
        #
        #     for device_id in range(torch.cuda.device_count()):
        #         pool_worker_id = device_id
        #         if mp.current_process().name == 'PoolWorker-' + str(pool_worker_id):
        #             device = 'cuda:' + str(device_id)
        #             break
        #
        # except RuntimeError as e:
        #     # if no device is available
        #     print(e)
        #     pass

    return device


def adni_extract_from_file_name(file_name):
    import re
    # file_name = 'sub-ADNI002S0729_ses-M06.vtk'
    m = re.search('\Asub-ADNI(.+?)_ses-M(.+?).vtk', file_name)
    if m:
        assert len(m.groups()) == 2
        subject_id = m.group(1)
        visit_age = m.group(2)
        return subject_id, visit_age
    else:
        raise LookupError('could not extract id and age from ' + file_name)