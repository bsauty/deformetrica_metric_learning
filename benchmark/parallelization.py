#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import gc

from api import Deformetrica
import support.utilities as utilities
from unit_tests import example_data_dir
import os
import time


data_dir = os.path.join(os.path.dirname(__file__), "data")


# dataset_specifications = {
#     'dataset_filenames': [
#         [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala1.vtk',
#           'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo1.vtk'}],
#         [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala2.vtk',
#           'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo2.vtk'}],
#         [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala3.vtk',
#           'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo3.vtk'}],
#         [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala4.vtk',
#           'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo4.vtk'}]],
#     'subject_ids': ['subj1', 'subj2', 'subj3', 'subj4']
# }
# template_specifications = {
#     'amygdala': {'deformable_object_type': 'SurfaceMesh',
#                  'kernel_type': 'torch', 'kernel_width': 15.0,
#                  'noise_std': 10.0,
#                  'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amyg_prototype.vtk',
#                  'attachment_type': 'varifold'},
#     'hippo': {'deformable_object_type': 'SurfaceMesh',
#               'kernel_type': 'torch', 'kernel_width': 15.0,
#               'noise_std': 6.0,
#               'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo_prototype.vtk',
#               'attachment_type': 'varifold'}
# }


dataset_specifications = {'dataset_filenames': [], 'subject_ids': []}
for file in os.listdir(data_dir + '/landmark/3d/right_hippocampus_2738'):
    subject_id, visit_age = utilities.adni_extract_from_file_name(file)

    dataset_specifications['dataset_filenames'].append(
        [{'hippo': data_dir + '/landmark/3d/right_hippocampus_2738/' + file}],
    )
    dataset_specifications['subject_ids'].append(subject_id)

template_specifications = {
    'hippo': {'deformable_object_type': 'SurfaceMesh',
              'kernel_type': 'torch', 'kernel_width': 15.0,
              'noise_std': 6.0,
              'filename': data_dir + '/landmark/3d/right_hippocampus_2738/sub-ADNI002S0729_ses-M00.vtk',
              'attachment_type': 'varifold'}
}


def deterministic_atlas_3d_brain_structure(nb_process):
    with Deformetrica(verbosity='DEBUG') as deformetrica:
        deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 3},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 7.0,
                           'number_of_threads': nb_process},
            write_output=False)


RUN_CONFIG = [
    # (deterministic_atlas_3d_brain_structure, 1),
    (deterministic_atlas_3d_brain_structure, 2),
    (deterministic_atlas_3d_brain_structure, 3),
    (deterministic_atlas_3d_brain_structure, 4),
    (deterministic_atlas_3d_brain_structure, 6),
    (deterministic_atlas_3d_brain_structure, 8)
]


if __name__ == "__main__":

    nb_processes = []
    results = []

    for current_run_config in RUN_CONFIG:
        func, nb_process = current_run_config
        print('>>>>>>>>>>>>> func=' + str(func) + ', nb_process=' + str(nb_process))

        start = time.perf_counter()
        func(nb_process)
        elapsed_time = time.perf_counter()-start
        print('run time: ' + str(elapsed_time))

        nb_processes.append(nb_process)
        results.append(elapsed_time)

        # cleanup between runs
        time.sleep(0.5)
        gc.collect()
        time.sleep(0.5)

    print('===== RESULTS =====')
    print(results)

    # assert len(nb_processes) == len(results)
    #
    # index = np.arange(len(RUN_CONFIG))
    # bar_width = 0.2
    # opacity = 0.4
    #
    # fig, ax = plt.subplots()
    #
    # ax.bar(index + bar_width, results, bar_width, label=':')
    #
    # ax.set_xlabel('Nb process')
    # ax.set_ylabel('Runtime (s)')
    # ax.set_title('Runtime by number of processes')
    # # ax.set_xticks(index + bar_width * ((len(kernels)*len(initial_devices))/2) - bar_width/2)
    # # ax.set_xticklabels([r['setup']['tensor_size'] for r in results if r['setup']['device'] == 'cpu'])
    # ax.legend()
    #
    # fig.tight_layout()
    #
    # plt.show()
