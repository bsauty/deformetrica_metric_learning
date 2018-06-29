from pprint import pprint
import os

import support.kernels as kernel_factory
from api.deformetrica import Deformetrica
from core.estimators.gradient_ascent import GradientAscent
from core.estimators.scipy_optimize import ScipyOptimize
from in_out.dataset_functions import create_dataset
from api.deformetrica import Deformetrica

def call(name, params):
    globals()[name](**params)
    
    

corresp = {
    
    "kernel": {
        "torch": kernel_factory.Type.TORCH,
        "keops": kernel_factory.Type.KEOPS
    }
    
}

def estimate_deterministic_atlas(deformation_parameters, template, optimization_parameters):
    
    deformetrica = Deformetrica(output_dir=os.path.join(os.path.dirname(__file__), 'output'))
    
    visit_ages = []
    subject_ids = []
    template_specifications = {}
    dataset_file_names = []
    
    i=0
    subjCount = len(template[0]["filenames"])
    for temp in template:
        rtemp = dict(temp)
        rtemp["kernel"] = kernel_factory.factory(corresp["kernel"][temp["kernel_type"]], kernel_width=temp["kernel_width"])
        subjCount = subjCount if len(temp["filenames"]) > subjCount else len(temp["filenames"])
        template_specifications[str(i)] = rtemp
        i+=1
    
    for subjID in range(subjCount):
        visit_ages.append([])
        subject_ids.append(str(subjID))
        obj = {}
        for j in range(i):
            obj[str(j)] = template_specifications[str(j)]["filenames"][subjID]
        dataset_file_names.append([obj])
        
    

    dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=template[0]["dimension"])
    
    deformetrica.estimate_deterministic_atlas(template_specifications,
                                              dataset,
                                              estimator=ScipyOptimize(max_iterations=10),
                                              deformation_kernel=kernel_factory.factory(corresp["kernel"][deformation_parameters["kernel_type"]], kernel_width=deformation_parameters["kernel_width"]),
                                              **optimization_parameters)

