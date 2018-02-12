import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings

from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel


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
        # If the cp or mom have been modified:
        self.shoot_is_modified = True
        # If the template data has been modified
        self.flow_is_modified = True
        # Wether to use a RK2 or a simple euler for shooting.
        self.use_rk2 = None
        # Norm of the deformation, lazily updated
        self.norm_squared = None
        # Contains the inverse kernel matrices for the time points 1 to self.number_of_time_points
        # (ACHTUNG does not contain the initial matrix, it is not needed)
        self.cometric_matrices = []

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2(self, use_rk2):
        self.shoot_is_modified = True
        self.use_rk2 = use_rk2

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_initial_template_data(self, td):
        self.initial_template_data = td
        self.flow_is_modified = True

    def get_initial_template_data(self):
        return self.initial_template_data

    def set_initial_template_data_from_numpy(self, td):
        td = Variable(torch.from_numpy(td).type(Settings().tensor_scalar_type))
        self.set_initial_template_data(td)

    def set_initial_control_points(self, cps):
        self.shoot_is_modified = True
        self.initial_control_points = cps

    def get_initial_control_points(self):
        return self.initial_control_points

    def set_initial_control_points_from_numpy(self, cps):
        cp = Variable(torch.from_numpy(cps).type(Settings().tensor_scalar_type))
        self.set_initial_control_points(cp)

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta(self, mom):
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta_from_numpy(self, mom):
        initial_mom = Variable(torch.from_numpy(mom).type(Settings().tensor_scalar_type))
        self.set_initial_momenta(initial_mom)

    def get_template_data(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            assert False, "You tried to get some template data, but the flow was modified, I advise updating the diffeo before getting this."
        if time_index is None:
            return self.template_data_t[- 1]
        return self.template_data_t[time_index]

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call shoot or flow on the deformation.
        """
        assert self.number_of_time_points > 0
        if self.shoot_is_modified:
            self.cometric_matrices = []
            self._shoot()
            self.shoot_is_modified = False
            if self.initial_template_data is not None:
                self._flow()
                self.flow_is_modified = False
            else:
                msg = "In exponential update, I am not flowing because I don't have any template data to flow"
                warnings.warn(msg)

        if self.flow_is_modified:
            if self.initial_template_data is not None:
                self._flow()
                self.flow_is_modified = False
            else:
                msg = "In exponential update, I am not flowing because I don't have any template data to flow"
                warnings.warn(msg)

    def parallel_transport(self, momenta_to_transport, with_tangential_component=True):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        """

        # Sanity check:
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert (momenta_to_transport.size() == self.initial_momenta.size())

        # Special cases, where the transport is simply the identity:
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).data.numpy()[0] < 1e-15 or
                    torch.norm(momenta_to_transport).data.numpy()[0] < 1e-15):
            parallel_transport_t = [momenta_to_transport] * self.number_of_time_points
            return parallel_transport_t

        # Initialize an exact kernel
        kernel = create_kernel('exact', self.kernel.kernel_width)

        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # First, get the scalar product initial_momenta \cdot momenta_to_transport and project momenta_to_transport
        # onto the orthogonal of initial_momenta.
        sp = torch.dot(momenta_to_transport,
                       kernel.convolve(self.initial_control_points, self.initial_control_points,
                                       self.initial_momenta)) / self.get_norm_squared()
        momenta_to_transport_orthogonal = momenta_to_transport - sp * self.initial_momenta

        sp_for_assert = torch.dot(
            momenta_to_transport_orthogonal, kernel.convolve(
                self.initial_control_points, self.initial_control_points, self.initial_momenta)).data.numpy()[0] \
                        / self.get_norm_squared().data.numpy()[0]
        assert sp_for_assert < 1e-5, "Projection onto orthogonal not orthogonal {e}".format(e=sp_for_assert)

        # Then, store the norm of this orthogonal momenta.
        initial_norm_squared = torch.dot(momenta_to_transport_orthogonal, kernel.convolve(
            self.initial_control_points, self.initial_control_points, momenta_to_transport_orthogonal))

        parallel_transport_t = [momenta_to_transport_orthogonal]

        for i in range(self.number_of_time_points - 1):
            # Shoot the two perturbed geodesics ------------------------------------------------------------------------
            cp_eps_pos = self._rk2_step(self.control_points_t[i],
                                        self.momenta_t[i] + epsilon * parallel_transport_t[-1], h, return_mom=False)
            cp_eps_neg = self._rk2_step(self.control_points_t[i],
                                        self.momenta_t[i] - epsilon * parallel_transport_t[-1], h, return_mom=False)

            # Compute J/h ----------------------------------------------------------------------------------------------
            approx_velocity = (cp_eps_pos - cp_eps_neg) / (2 * epsilon * h)

            # We need to find the cotangent space version of this vector -----------------------------------------------
            # If we don't have already the cometric matrix, we compute and store it.
            # TODO: add optionnal flag for not saving this if it's too large.
            if not len(self.cometric_matrices) == self.number_of_time_points - 1:
                kernel_matrix = kernel.get_kernel_matrix(self.control_points_t[i + 1])
                self.cometric_matrices.append(torch.inverse(kernel_matrix))

            # Solve the linear system.
            approx_momenta = torch.mm(self.cometric_matrices[i], approx_velocity)

            # We get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = torch.dot(approx_momenta, kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], self.momenta_t[i + 1])) \
                                        / self.get_norm_squared()

            approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

            # Renormalization ------------------------------------------------------------------------------------------
            approx_momenta_norm_squared = torch.dot(approx_momenta, kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], approx_momenta))
            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.data.numpy()[0] - 1.) > 0.02:
                msg = "Watch out, a large renormalization factor %.4f is required during the parallel transport, " \
                      "please use a finer discretization." \
                      % (approx_momenta_norm_squared.data.numpy()[0] / initial_norm_squared.data.numpy()[0])
                warnings.warn(msg)

            # Finalization ---------------------------------------------------------------------------------------------
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == len(self.momenta_t), "Oups, something went wrong."

        # We now need to add back the component along the velocity to the transported vectors.
        if with_tangential_component:
            parallel_transport_t = \
                [parallel_transport_t[i] + sp * self.momenta_t[i] for i in range(self.number_of_time_points)]

        return parallel_transport_t

    def get_norm_squared(self):
        if self.shoot_is_modified:
            msg = "Watch out, you are getting the norm of the deformation, but the shoot was modified without " \
                  "updating, I should probably throw an error for this..."
            warnings.warn(msg)
        return self.norm_squared

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _shoot(self):
        """
        Computes the flow of momenta and control points
        """
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"

        # Special case, with nearly zero initial momenta.
        if torch.norm(self.initial_momenta).data.cpu().numpy()[0] < 1e-15:
            self.control_points_t = [self.initial_control_points] * self.number_of_time_points
            self.momenta_t = [self.initial_momenta] * self.number_of_time_points

        # Otherwise, integrate the Hamiltonian equations.
        else:
            self.control_points_t = []
            self.momenta_t = []
            self.control_points_t.append(self.initial_control_points)
            self.momenta_t.append(self.initial_momenta)
            dt = 1.0 / float(self.number_of_time_points - 1)
            for i in range(self.number_of_time_points - 1):
                if self.use_rk2:
                    new_cp, new_mom = self._rk2_step(self.control_points_t[i], self.momenta_t[i], dt, return_mom=True)
                else:
                    new_cp, new_mom = self._euler_step(self.control_points_t[i], self.momenta_t[i], dt)

                self.control_points_t.append(new_cp)
                self.momenta_t.append(new_mom)

        # Updating the squared norm attribute.
        self.update_norm_squared()

    def _flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert not self.shoot_is_modified, "CP or momenta were modified and the shoot not computed, and now you are asking me to flow ?"
        assert len(self.control_points_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"

        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_data_t = []
        self.template_data_t.append(self.initial_template_data)
        for i in range(self.number_of_time_points - 1):
            d_pos = self.kernel.convolve(self.template_data_t[i], self.control_points_t[i], self.momenta_t[i])
            self.template_data_t.append(self.template_data_t[i] + dt * d_pos)

            if self.use_rk2:
                # In this case improved euler (= Heun's method) to save one computation of convolve gradient.
                self.template_data_t[-1] = self.template_data_t[i] + dt / 2 * (self.kernel.convolve(
                    self.template_data_t[-1], self.control_points_t[i + 1], self.momenta_t[i + 1]) + d_pos)

    def update_norm_squared(self):
        self.norm_squared = torch.dot(self.initial_momenta.view(-1), self.kernel.convolve(
            self.initial_control_points, self.initial_control_points, self.initial_momenta).view(-1))

    def _euler_step(self, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        return cp + h * self.kernel.convolve(cp, cp, mom), mom - h * self.kernel.convolve_gradient(mom, cp)

    def _rk2_step(self, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        mid_cp = cp + h / 2. * self.kernel.convolve(cp, cp, mom)
        mid_mom = mom - h / 2. * self.kernel.convolve_gradient(mom, cp)
        if return_mom:
            return cp + h * self.kernel.convolve(mid_cp, mid_cp, mid_mom), mom - h * \
                   self.kernel.convolve_gradient(mid_mom, mid_cp)
        else:
            return cp + h * self.kernel.convolve(mid_cp, mid_cp, mid_mom)

    def parallel_transport(self, momenta_to_transport, with_tangential_component=True):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        """

        # Sanity check:
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert (momenta_to_transport.size() == self.initial_momenta.size())

        # Special cases, where the transport is simply the identity:
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).data.numpy()[0] < 1e-15 or
                    torch.norm(momenta_to_transport).data.numpy()[0] < 1e-15):
            parallel_transport_t = [momenta_to_transport] * self.number_of_time_points
            return parallel_transport_t

        # Initialize an exact kernel
        kernel = create_kernel('exact', self.kernel.kernel_width)

        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # First, get the scalar product initial_momenta \cdot momenta_to_transport and project momenta_to_transport
        # onto the orthogonal of initial_momenta.
        sp = torch.dot(momenta_to_transport,
                       kernel.convolve(self.initial_control_points, self.initial_control_points,
                                       self.initial_momenta)) / self.get_norm_squared()
        momenta_to_transport_orthogonal = momenta_to_transport - sp * self.initial_momenta

        sp_for_assert = torch.dot(
            momenta_to_transport_orthogonal, kernel.convolve(
                self.initial_control_points, self.initial_control_points, self.initial_momenta)).data.numpy()[0] \
                        / self.get_norm_squared().data.numpy()[0]
        assert sp_for_assert < 1e-5, "Projection onto orthogonal not orthogonal {e}".format(e=sp_for_assert)

        # Then, store the norm of this orthogonal momenta.
        initial_norm_squared = torch.dot(momenta_to_transport_orthogonal, kernel.convolve(
            self.initial_control_points, self.initial_control_points, momenta_to_transport_orthogonal))

        parallel_transport_t = [momenta_to_transport_orthogonal]

        for i in range(self.number_of_time_points - 1):
            # Shoot the two perturbed geodesics ------------------------------------------------------------------------
            cp_eps_pos = self._rk2_step(self.control_points_t[i],
                                        self.momenta_t[i] + epsilon * parallel_transport_t[-1], h, return_mom=False)
            cp_eps_neg = self._rk2_step(self.control_points_t[i],
                                        self.momenta_t[i] - epsilon * parallel_transport_t[-1], h, return_mom=False)

            # Compute J/h ----------------------------------------------------------------------------------------------
            approx_velocity = (cp_eps_pos - cp_eps_neg) / (2 * epsilon * h)

            # We need to find the cotangent space version of this vector -----------------------------------------------
            # If we don't have already the cometric matrix, we compute and store it.
            # TODO: add optionnal flag for not saving this if it's too large.
            if not len(self.cometric_matrices) == self.number_of_time_points - 1:
                kernel_matrix = kernel.get_kernel_matrix(self.control_points_t[i + 1])
                self.cometric_matrices.append(torch.inverse(kernel_matrix))

            # Solve the linear system.
            approx_momenta = torch.mm(self.cometric_matrices[i], approx_velocity)

            # We get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = torch.dot(approx_momenta, kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], self.momenta_t[i + 1])) \
                                        / self.get_norm_squared()

            approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

            # Renormalization ------------------------------------------------------------------------------------------
            approx_momenta_norm_squared = torch.dot(approx_momenta, kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], approx_momenta))
            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.data.numpy()[0] - 1.) > 0.02:
                msg = ("Watch out, a large renormalization factor %.4f is required during the parallel transport, "
                       "please use a finer discretization."
                       % (approx_momenta_norm_squared.data.numpy()[0] / initial_norm_squared.data.numpy()[0]))
                warnings.warn(msg)

            # Finalization ---------------------------------------------------------------------------------------------
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == len(self.momenta_t), "Oups, something went wrong."

        # We now need to add back the component along the velocity to the transported vectors.
        if with_tangential_component:
            parallel_transport_t = \
                [parallel_transport_t[i] + sp * self.momenta_t[i] for i in range(self.number_of_time_points)]

        return parallel_transport_t

    def write_flow(self, objects_names, objects_extensions, template, write_adjoint_parameters=False):
        assert (not (
            self.flow_is_modified)), "You are trying to write data relative to the flow, but it has been modified and not updated."
        for j, data in enumerate(self.template_data_t):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names):
                names.append(elt + "__tp_" + str(j) + objects_extensions[k])
            aux_points = template.get_points()
            template.set_data(data.data.numpy())
            template.write(names)
            # restauring state of the template object for further computations
            template.set_data(aux_points)
            # saving control points and momenta
            cp = self.control_points_t[j].data.numpy()
            mom = self.momenta_t[j].data.numpy()

            if write_adjoint_parameters:
                write_2D_array(cp, elt + "__ControlPoints__tp_" + str(j) + ".txt")
                write_3D_array(mom, elt + "__Momenta__tp_" + str(j) + ".txt")
            # write_control_points_and_momenta_vtk(cp, mom, elt + "_mom_and_cp_" + str(j) + ".vtk")

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert (not (
            self.shoot_is_modified)), "You are trying to write data relative to the shooting, but it has been modified and not updated."
        assert len(self.control_points_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        for j, (control_points, momenta) in enumerate(zip(self.control_points_t, self.momenta_t)):
            write_2D_array(control_points.data.numpy(), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(momenta.data.numpy(), name + "__momenta_" + str(j) + ".txt")
            write_control_points_and_momenta_vtk(control_points.data.numpy(), momenta.data.numpy(),
                                                 name + "_momenta_and_control_points_" + str(j) + ".vtk")
