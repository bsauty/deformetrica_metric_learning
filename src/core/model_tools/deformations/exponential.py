import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
import torch
from torch.autograd import Variable
import warnings

class Exponential:
    """
    Control-point-based LDDMM exponential, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.kernel = None
        self.number_of_time_points = None
        # Initial position of control points
        self.initial_control_points = None
        # Control points trajectory
        self.control_points_t = None
        # Initial momenta
        self.initial_momenta = None
        # Momenta trajectory
        self.momenta_t = None
        # Initial template data
        self.initial_template_data = None
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_data_t = None
        #If the cp or mom have been modified:
        self.shoot_is_modified = True
        #If the template data has been modified
        self.flow_is_modified = True

        # Contains the cholesky decomp of the kernel matrices
        # for the time points 1 to self.number_of_time_points
        # (ACHTUNG does not contain the decomp of the initial kernel matrix, it is not needed)
        self.cholesky_kernel_matrices = []


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_template_data(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            assert False, "You tried to get some template data, but the flow was modified, I advise updating the diffeo before getting this."
        if time_index is None:
            return self.template_data_t[- 1]
        return self.template_data_t[time_index]

    def set_initial_control_points(self, cps):
        """
        Set the control points and the ismodified flag. Torch input is assumed
        """
        self.shoot_is_modified = True
        self.initial_control_points = cps

    def set_initial_momenta(self, mom):
        """
        Set the initial momenta and the ismodified flag. Torch input is assumed
        """
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def set_initial_template_data(self, td):
        self.initial_template_data = td
        self.flow_is_modified = True

    def set_initial_control_points_from_numpy(self, cps):
        """
        set_initial_control_points from numpy arg
        """
        cp = Variable(torch.from_numpy(cps).type(Settings().tensor_scalar_type))
        self.set_initial_control_points(cp)

    def set_initial_template_data_from_numpy(self, td):
        """
        set_template_data from numpy arg
        """
        td = Variable(torch.from_numpy(td).type(Settings().tensor_scalar_type))
        self.set_initial_template_data(td)

    def set_initial_momenta_from_numpy(self, mom):
        """
        set_initial_momenta from numpy arg
        """
        initial_mom = Variable(torch.from_numpy(mom).type(Settings().tensor_scalar_type))
        self.set_initial_momenta(initial_mom)


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call shoot or flow on the deformation.
        """
        assert self.number_of_time_points > 0
        if self.shoot_is_modified:
            self.cholesky_kernel_matrices = []
            self._shoot()
            self.shoot_is_modified = False
            self._flow()
            self.flow_is_modified = False

        if self.flow_is_modified:
            self._flow()
            self.flow_is_modified  = False


    def get_norm_squared(self):
        assert not(self.shoot_is_modified), "Can't get norm of modified object without updating."
        return self.norm_squared

    # Write functions --------------------------------------------------------------------------------------------------
    def write_flow(self, objects_names, objects_extensions, template):
        assert (not(self.flow_is_modified)), "You are trying to write data relative to the flow, but it has been modified and not updated."
        for j, data in enumerate(self.template_data_t):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names): names.append(elt + "_t=" + str(j) + objects_extensions[k])
            aux_points = template.get_data()
            template.set_data(data.data.numpy())
            template.write(names)
            # restauring state of the template object for further computations
            template.set_data(aux_points)

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert (not(self.shoot_is_modified)), "You are trying to write data relative to the shooting, but it has been modified and not updated."
        assert len(self.control_points_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        for j, (control_points, momenta) in enumerate(zip(self.control_points_t, self.momenta_t)):
            write_2D_array(control_points.data.numpy(), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(momenta.data.numpy(), name + "__momenta_" + str(j) + ".txt")

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _shoot(self):
        """
        Computes the flow of momenta and control points
        """
        # TODO : not shoot if small momenta norm
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.PositionsT = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        #     self.InitialMomenta = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        self.control_points_t = []
        self.momenta_t = []
        self.control_points_t.append(self.initial_control_points)
        self.momenta_t.append(self.initial_momenta)
        dt = 1.0 / float(self.number_of_time_points - 1)
        # REPLACE with an hamiltonian (e.g. une classe hamiltonien)
        for i in range(self.number_of_time_points - 1):
            dPos = self.kernel.convolve(self.control_points_t[i], self.control_points_t[i], self.momenta_t[i])
            dMom = self.kernel.convolve_gradient(self.momenta_t[i], self.control_points_t[i])
            self.control_points_t.append(self.control_points_t[i] + dt * dPos)
            self.momenta_t.append(self.momenta_t[i] - dt * dMom)
        #We now set the norm_squared attribute of the exponential:
        self.norm_squared = torch.dot(self.initial_momenta.view(-1), self.kernel.convolve(
            self.initial_control_points, self.initial_control_points, self.initial_momenta).view(-1))

    def _flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert (not(self.shoot_is_modified)), "CP or momenta were modified and the shoot not computed, and now you are asking me to flow ?"
        assert len(self.control_points_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"
        assert len(self.initial_template_data) > 0, "Please give landmark points to flow"

        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_data_t = []
        self.template_data_t.append(self.initial_template_data)
        for i in range(self.number_of_time_points - 1):
            dPos = self.kernel.convolve(self.template_data_t[i], self.control_points_t[i], self.momenta_t[i])
            self.template_data_t.append(self.template_data_t[i] + dt * dPos)

    def parallel_transport(self, momenta_to_transport):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        """

        #Sanity check:
        assert(not(self.shoot_is_modified)), "You want to parallel transport but the shoot was modified, update please."
        assert (momenta_to_transport.size() == self.initial_momenta.size())


        #Initialize an exact kernel #TODO : use any kernel here (requires some tricks though)
        kernel = create_kernel('exact', self.kernel.kernel_width)

        h = 1./(self.number_of_time_points - 1.)
        epsilon = h

        #First, get the scalar product initial_momenta \cdot momenta_to_transport and project momenta_to_transport onto the orthogonal of initial_momenta
        sp = torch.dot(momenta_to_transport, kernel.convolve(self.initial_control_points, self.initial_control_points, self.initial_momenta)) / self.get_norm_squared()
        momenta_to_transport_orth = momenta_to_transport - sp * self.initial_momenta
        assert torch.dot(momenta_to_transport, kernel.convolve(self.initial_control_points, self.initial_control_points, self.initial_momenta)).data.numpy()[0] < 1e-5, "Projection onto orthogonal not orthogonal !"
        initial_norm = torch.dot(momenta_to_transport_orth, kernel.convolve(self.initial_control_points, self.initial_control_points, momenta_to_transport_orth))

        #So now we only transport momenta_to_transport_orth, and keep its norm constant, we'll stitch the non ortho component in the end
        parallel_transport_t = [momenta_to_transport_orth]

        def rk2_step(cp, mom):
            """
            perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
            """
            dpos1 = kernel.convolve(cp, cp, mom)
            dmom1 = kernel.convolve_gradient(mom, cp)
            cp1 = cp + h/2. * dpos1
            mom2 = mom - h/2. * dmom1
            dpos2 = kernel.convolve(cp1, cp1, mom2)
            return cp + h * dpos2

        for i in range(self.number_of_time_points - 1):
            #Shoot the two perturbed geodesics
            cp_eps_pos = rk2_step(self.control_points_t[i], self.momenta_t[i] + epsilon * parallel_transport_t[-1])
            cp_eps_neg = rk2_step(self.control_points_t[i], self.momenta_t[i] - epsilon * parallel_transport_t[-1])

            #Compute J/h and
            approx_velocity = (cp_eps_pos-cp_eps_neg)/(2 * epsilon * h)

            #We need to find the cotangent space version of this vector
            #First case: we already have the cholesky decomposition of the kernel matrix, we use it:
            if len(self.cholesky_kernel_matrices) == self.number_of_time_points - 1:
                approx_momenta = torch.potrs(approx_velocity, self.cholesky_kernel_matrices[i])

            # Second case: we don't have the cholesky decomposition: we compute and store it (#TODO: add optionnal flag for not saving this if it's too large)
            else:
                kernel_matrix = kernel.get_kernel_matrix(self.control_points_t[i+1])
                cholesky_kernel_matrix = torch.potrf(kernel_matrix)
                self.cholesky_kernel_matrices.append(cholesky_kernel_matrix)
                approx_momenta = torch.potrs(approx_velocity, cholesky_kernel_matrix).squeeze()

            #we get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = torch.dot(approx_momenta, kernel.convolve(self.control_points_t[i+1], self.control_points_t[i+1], self.momenta_t[i+1])) / self.get_norm_squared()
            print("Scalar prof with velocity :", scalar_prod_with_velocity.data.numpy()[0])
            approx_momenta -= scalar_prod_with_velocity * self.momenta_t[i+1]

            norm_approx_momenta = torch.dot(approx_momenta, kernel.convolve(self.control_points_t[i+1], self.control_points_t[i+1], approx_momenta))

            if (abs(norm_approx_momenta.data.numpy()[0]/initial_norm.data.numpy()[0] - 1.) > 0.02):
                msg = "Watch out, a large renormalization (factor {f} is required during the parallel transport, please use a finer discretization.".format(f=norm_approx_momenta.data.numpy()[0]/initial_norm.data.numpy()[0])
                warning.warn(msg)

            #Renormalizing this component.
            renorm_momenta = approx_momenta * initial_norm / norm_approx_momenta

            parallel_transport_t.append(renorm_momenta)

        assert len(parallel_transport_t) == len(self.momenta_t), "Oups, something went wrong."

        #We now need to add back the component along the velocity to the transported vectors.
        parallel_transport_t = [parallel_transport_t[i] + sp * self.momenta_t[i] for i in range(self.number_of_time_points)]

        return parallel_transport_t














