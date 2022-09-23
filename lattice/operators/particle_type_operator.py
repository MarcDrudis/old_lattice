import copy


class ParticleTypeOperator:
    """
    A parent class for all Operators with a particle type (e.g. FermionicOperator, SpinOperator, etc.) on a lattice.
    """

    def __init__(self, particle_type):
        self._particle_type = particle_type

    @property
    def particle_type(self):
        """Return the particle type"""
        return copy.deepcopy(self._particle_type)

    def __matmul__(self, other):
        """
        Implements the tensor product for `ParticleTypeOperator` objects. The tensor product order is [self, other].

        Args:
            other (ParticleTypeOperator/MixedOperator):


        Returns:
            MixedOperator
        """
        from .mixed_operator import MixedOperator

        """Implements the operator tensorproduct"""
        if isinstance(other, ParticleTypeOperator):
            assert other.particle_type != self.particle_type, \
                "You are trying to tensor together two '{0}' type registers. Please include all '{0}' operators " \
                "into one single register.".format(other.particle_type)
            return MixedOperator([self, other])

        elif isinstance(other, MixedOperator):
            new_mixed_operator = copy.deepcopy(other)
            assert self.particle_type not in new_mixed_operator.registers, \
                "Operator already has a '{0}' register. Please include all '{0}' operators " \
                "into this register.".format(self.particle_type)
            new_mixed_operator[self.particle_type] = self
            new_mixed_operator._registers = [self.particle_type] + other.registers
            return new_mixed_operator

        else:
            raise TypeError("unsupported operand @ for objects of type '{}' and '{}'".format(type(self).__name__,
                                                                                             type(other).__name__))
