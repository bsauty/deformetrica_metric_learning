import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings

from pydeformetrica.src.in_out.array_readers_and_writers import *
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
        # Initial template points
        self.initial_template_points = None
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_points_t = None
        # If the cp or mom have been modified:
        self.shoot_is_modified = True
        # If the template points has been modified
        self.flow_is_modified = True
        # Wether to use a RK2 or a simple euler for shooting.
        self.use_rk2 = None
        # Norm of the deformation, lazily updated
        self.norm_squared = None
        # Contains the inverse kernel matrices for the time points 1 to self.number_of_time_points
        # (ACHTUNG does not contain the initial matrix, it is not needed)
        self.cometric_matrices = {}

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2(self, use_rk2):
        self.shoot_is_modified = True
        self.use_rk2 = use_rk2

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_initial_template_points(self, td):
        self.initial_template_points = td
        self.flow_is_modified = True

    def get_initial_template_points(self):
        return self.initial_template_points

    def set_initial_control_points(self, cps):
        self.shoot_is_modified = True
        self.initial_control_points = cps

    def get_initial_control_points(self):
        return self.initial_control_points

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta(self, mom):
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def get_initial_momenta(self):
        return self.initial_momenta

    def get_template_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            assert False, "You tried to get some template points, but the flow was modified, I advise updating the diffeo before getting this."
        if time_index is None:
            return self.template_points_t[- 1]
        return self.template_points_t[time_index]

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
            self.cometric_matrices = {}
            self.shoot()
            self.shoot_is_modified = False
            if self.initial_template_points is not None:
                self.flow()
                self.flow_is_modified = False
            elif not Settings().dense_mode:
                msg = "In exponential update, I am not flowing because I don't have any template points to flow"
                warnings.warn(msg)

        if self.flow_is_modified:
            if self.initial_template_points is not None:
                self.flow()
                self.flow_is_modified = False
            elif not Settings().dense_mode:
                msg = "In exponential update, I am not flowing because I don't have any template points to flow"
                warnings.warn(msg)

    def parallel_transport(self, momenta_to_transport, initial_time_point=0,
                           with_tangential_component=True, orthogonalize=True):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        """

        # Sanity checks ------------------------------------------------------------------------------------------------
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert (momenta_to_transport.size() == self.initial_momenta.size())

        # Special cases, where the transport is simply the identity ----------------------------------------------------
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        # if (torch.norm(self.initial_momenta).data.cpu().numpy()[0] < 1e-15 or
        #             torch.norm(momenta_to_transport).data.cpu().numpy()[0] < 1e-15):
        #     parallel_transport_t = [momenta_to_transport] * self.number_of_time_points
        #     return parallel_transport_t

        # Step sizes ---------------------------------------------------------------------------------------------------
        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # Optional initial orthogonalization ---------------------------------------------------------------------------
        if orthogonalize:
            sp = torch.dot(momenta_to_transport,
                           self.kernel.convolve(
                               self.control_points_t[initial_time_point], self.control_points_t[initial_time_point],
                               self.momenta_t[initial_time_point])) / self.get_norm_squared()
            momenta_to_transport_orthogonal = momenta_to_transport - sp * self.momenta_t[initial_time_point]

            sp_for_assert = torch.dot(
                momenta_to_transport_orthogonal, self.kernel.convolve(
                    self.control_points_t[initial_time_point], self.control_points_t[initial_time_point],
                    self.momenta_t[initial_time_point])).data.cpu().numpy()[0] \
                            / self.get_norm_squared().data.cpu().numpy()[0]
            assert sp_for_assert < 1e-4, "Projection onto orthogonal not orthogonal {e}".format(e=sp_for_assert)

            parallel_transport_t = [momenta_to_transport_orthogonal]

        else:
            parallel_transport_t = [momenta_to_transport]

        # Then, store the initial norm of this orthogonal momenta ------------------------------------------------------
        initial_norm_squared = torch.dot(parallel_transport_t[0], self.kernel.convolve(
            self.control_points_t[initial_time_point], self.control_points_t[initial_time_point],
            parallel_transport_t[0]))

        for i in range(initial_time_point, self.number_of_time_points - 1):
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
            if i not in self.cometric_matrices:
                kernel_matrix = self.kernel.get_kernel_matrix(self.control_points_t[i + 1])
                self.cometric_matrices[i] = torch.inverse(kernel_matrix)

            # Solve the linear system.
            approx_momenta = torch.mm(self.cometric_matrices[i], approx_velocity)

            # We get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = torch.dot(approx_momenta, self.kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], self.momenta_t[i + 1])) \
                                        / self.get_norm_squared()

            approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

            # Renormalization ------------------------------------------------------------------------------------------
            approx_momenta_norm_squared = torch.dot(approx_momenta, self.kernel.convolve(
                self.control_points_t[i + 1], self.control_points_t[i + 1], approx_momenta))
            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.data.cpu().numpy()[0] - 1.) > 0.5:
                raise ValueError('Absurd required renormalization factor during parallel transport. Exception raised.')
            elif abs(renormalization_factor.data.cpu().numpy()[0] - 1.) > 0.02:
                msg = ("Watch out, a large renormalization factor %.4f is required during the parallel transport, "
                       "please use a finer discretization." % renormalization_factor.data.cpu().numpy()[0])
                warnings.warn(msg)

            # Finalization ---------------------------------------------------------------------------------------------
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == self.number_of_time_points - initial_time_point, \
            "Oops, something went wrong."

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
    ### Extension methods:
    ####################################################################################################################

    def extend(self, number_of_additional_time_points):

        # Special case of the exponential reduced to a single point.
        if self.number_of_time_points == 1:
            self.number_of_time_points += number_of_additional_time_points
            self.update()
            return

        # Extended shoot.
        dt = 1.0 / float(self.number_of_time_points - 1)  # Same time-step.
        for i in range(number_of_additional_time_points):
            if self.use_rk2:
                new_cp, new_mom = self._rk2_step(self.control_points_t[-1], self.momenta_t[-1], dt, return_mom=True)
            else:
                new_cp, new_mom = self._euler_step(self.control_points_t[-1], self.momenta_t[-1], dt)

            self.control_points_t.append(new_cp)
            self.momenta_t.append(new_mom)

        # Scaling of the new length.
        length_ratio = float(self.number_of_time_points + number_of_additional_time_points - 1) \
                       / float(self.number_of_time_points - 1)
        self.number_of_time_points += number_of_additional_time_points
        self.initial_momenta *= length_ratio
        self.momenta_t = [elt * length_ratio for elt in self.momenta_t]
        self.norm_squared *= length_ratio ** 2

        # Extended flow.
        # Special case of the dense mode.
        if Settings().dense_mode:
            self.template_points_t = self.control_points_t
            return

        # Standard case.
        for i in range(number_of_additional_time_points):
            d_pos = self.kernel.convolve(self.template_points_t[-1], self.control_points_t[-1], self.momenta_t[-1])
            self.template_points_t.append(self.template_points_t[-1] + dt * d_pos)

            if self.use_rk2:
                # In this case improved euler (= Heun's method) to save one computation of convolve gradient.
                self.template_points_t[-1] = self.template_points_t[-2] + dt / 2 * (self.kernel.convolve(
                    self.template_points_t[-1], self.control_points_t[-1], self.momenta_t[-1]) + d_pos)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def shoot(self):
        """
        Computes the flow of momenta and control points.
        """
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"

        # Integrate the Hamiltonian equations.
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

    def flow(self):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        assert not self.shoot_is_modified, "CP or momenta were modified and the shoot not computed, and now you are asking me to flow ?"
        assert len(self.control_points_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"

        # Special case of the dense mode.
        if Settings().dense_mode:
            assert 'image_points' not in self.initial_template_points.keys(), 'Dense mode not allowed with image data.'
            self.template_points_t = self.control_points_t
            return

        # Initialization.
        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_points_t = []
        for t in range(self.number_of_time_points):
            self.template_points_t.append(self.initial_template_points)

        # Flow landmarks points.
        if 'landmark_points' in self.initial_template_points.keys():
            for i in range(self.number_of_time_points - 1):
                d_pos = self.kernel.convolve(self.template_points_t[i]['landmark_points'],
                                             self.control_points_t[i], self.momenta_t[i])
                self.template_points_t[i + 1]['landmark_points'] \
                    = self.template_points_t[i]['landmark_points'] + dt * d_pos

                if self.use_rk2:
                    # In this case improved euler (= Heun's method) to save one computation of convolve gradient.
                    self.template_points_t[i + i]['landmark_points'] \
                        = self.template_points_t[i]['landmark_points'] + dt / 2 * (self.kernel.convolve(
                        self.template_points_t[-1], self.control_points_t[i + 1], self.momenta_t[i + 1]) + d_pos)

        # Flow image points.
        if 'image_points' in self.initial_template_points.keys():
            dimension = Settings().dimension
            image_shape = self.template_points_t[0]['image_points'].size()

            for i in range(self.number_of_time_points - 1):
                vf = self.kernel.convolve(self.template_points_t[0]['image_points'].contiguous().view(-1, dimension),
                                          self.control_points_t[i], self.momenta_t[i]).view(image_shape)
                dY = self._compute_image_explicit_euler_step_at_order_1(self.template_points_t[i]['image_points'], vf)
                self.template_points_t[i + 1]['image_points'] = self.template_points_t[i]['image_points'] - dY

            if self.use_rk2:
                msg = 'RK2 not implemented to flow image points.'
                warnings.warn(msg)

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
            return cp + h * self.kernel.convolve(mid_cp, mid_cp, mid_mom), \
                   mom - h * self.kernel.convolve_gradient(mid_mom, mid_cp)
        else:
            return cp + h * self.kernel.convolve(mid_cp, mid_cp, mid_mom)

    def write_flow(self, objects_names, objects_extensions, template, write_adjoint_parameters=False):
        assert (not (
            self.flow_is_modified)), "You are trying to write data relative to the flow, but it has been modified and not updated."
        for j, data in enumerate(self.template_points_t):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names):
                names.append(elt + "__tp_" + str(j) + objects_extensions[k])
            aux_points = template.get_points()
            template.set_points(data.data.numpy())
            template.write(names)
            # restoring state of the template object for further computations
            template.set_points(aux_points)
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

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    # TODO. Wrap pytorch of an efficient C code ? Use keops ? Called ApplyH in PyCa.
    def _compute_image_explicit_euler_step_at_order_1(self, Y, vf):
        dimension = Settings().dimension
        dY = Variable(torch.zeros(Y.shape).type(Settings().tensor_scalar_type))

        if dimension == 2:

            # X direction.
            for j in range(Y.shape[1]):

                # Top, i = 0 (forward).
                i = 0
                dY[i, j] = dY[i, j] - vf[i, j, 0] * Y[i, j]
                dY[i, j] = dY[i, j] + vf[i, j, 0] * Y[i + 1, j]

                # Core (central).
                for i in range(1, Y.shape[0] - 1):
                    dY[i, j] = dY[i, j] - 0.5 * vf[i, j, 0] * Y[i - 1, j]
                    dY[i, j] = dY[i, j] + 0.5 * vf[i, j, 0] * Y[i + 1, j]

                # Bottom, i = Y.shape[0] - 1 (backward).
                i = Y.shape[0] - 1
                dY[i, j] = dY[i, j] - vf[i, j, 0] * Y[i - 1, j]
                dY[i, j] = dY[i, j] + vf[i, j, 0] * Y[i, j]

            # Y direction.
            for i in range(Y.shape[0]):

                # Top, j = 0 (forward).
                j = 0
                dY[i, j] = dY[i, j] - vf[i, j, 1] * Y[i, j]
                dY[i, j] = dY[i, j] + vf[i, j, 1] * Y[i, j + 1]

                # Core (central).
                for j in range(1, Y.shape[1] - 1):
                    dY[i, j] = dY[i, j] - 0.5 * vf[i, j, 1] * Y[i, j - 1]
                    dY[i, j] = dY[i, j] + 0.5 * vf[i, j, 1] * Y[i, j + 1]

                # Bottom, j = Y.shape[1] - 1 (backward).
                j = Y.shape[1] - 1
                dY[i, j] = dY[i, j] - vf[i, j, 1] * Y[i, j - 1]
                dY[i, j] = dY[i, j] + vf[i, j, 1] * Y[i, j]

        elif dimension == 3:

            for k in range(Y.shape[2]):

                # X direction.
                for j in range(Y.shape[1]):

                    # Top, i = 0 (forward).
                    i = 0
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 0] * Y[i, j, k]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 0] * Y[i + 1, j, k]

                    # Core (central).
                    for i in range(1, Y.shape[0] - 1):
                        dY[i, j, k] = dY[i, j, k] - 0.5 * vf[i, j, k, 0] * Y[i - 1, j, k]
                        dY[i, j, k] = dY[i, j, k] + 0.5 * vf[i, j, k, 0] * Y[i + 1, j, k]

                    # Bottom, i = Y.shape[0] - 1 (backward).
                    i = Y.shape[0] - 1
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 0] * Y[i - 1, j, k]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 0] * Y[i, j, k]

                # Y direction.
                for i in range(Y.shape[0]):

                    # Top, j = 0 (forward).
                    j = 0
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 1] * Y[i, j, k]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 1] * Y[i, j + 1, k]

                    # Core (central).
                    for j in range(1, Y.shape[1] - 1):
                        dY[i, j, k] = dY[i, j, k] - 0.5 * vf[i, j, k, 1] * Y[i, j - 1, k]
                        dY[i, j, k] = dY[i, j, k] + 0.5 * vf[i, j, k, 1] * Y[i, j + 1, k]

                    # Bottom, j = Y.shape[1] - 1 (backward).
                    j = Y.shape[1] - 1
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 1] * Y[i, j - 1, k]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 1] * Y[i, j, k]

            # Z direction.
            for i in range(Y.shape[0]):
                for j in range(Y.range[1]):

                    # Top, k = 0 (forward).
                    k = 0
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 2] * Y[i, j, k]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 2] * Y[i, j, k + 1]

                    # Core (central).
                    for j in range(1, Y.shape[2] - 1):
                        dY[i, j, k] = dY[i, j, k] - 0.5 * vf[i, j, k, 2] * Y[i, j, k - 1]
                        dY[i, j, k] = dY[i, j, k] + 0.5 * vf[i, j, k, 2] * Y[i, j, k + 1]

                    # Bottom, j = Y.shape[2] - 1 (backward).
                    k = Y.shape[2] - 1
                    dY[i, j, k] = dY[i, j, k] - vf[i, j, k, 2] * Y[i, j, k - 1]
                    dY[i, j, k] = dY[i, j, k] + vf[i, j, k, 2] * Y[i, j, k]

        else:
            raise RuntimeError('Invalid dimension of the ambient space: %d' % dimension)

        return dY

