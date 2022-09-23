# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception for errors raised by the QISKit Lattice SDK."""


class QiskitLatticeError(Exception):
    """Base class for errors raised by the QiskitLattice SDK."""

    def __init__(self, *message):
        """Set the error message."""
        super(QiskitLatticeError, self).__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)