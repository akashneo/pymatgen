#!/usr/bin/env python

"""
This module implements a FloatWithUnit, which is a subclass of float. It
also defines supported units for some commonly used units for energy, length,
temperature, time and charge. FloatWithUnits also support conversion to one
another, and additions and subtractions perform automatic conversion if
units are detected.
"""

from __future__ import division

__author__ = "Shyue Ping Ong, Matteo Giantomassi"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Shyue Ping Ong, Matteo Giantomassi"
__status__ = "Production"
__date__ = "Aug 30, 2013"

import numpy as np

import collections
import numbers
from functools import partial
import math
import re

"""
Definitions of supported units. Values below are essentially scaling and
conversion factors. What matters is the relative values, not the absolute.
"""

#TODO: Define base units vs dervived units.
#length	meter	m
#mass	kilogram      	kg
#time	second	s
#electric current	ampere	A
#thermodynamic temperature      	kelvin	K
#amount of substance	mole	mol
#luminous intensity	candela	cd

#This current list is a mix of base and derived units.
SUPPORTED_UNITS = {
    "energy": {
        "eV": 1,
        "Ha": 27.21138386,
        "Ry": 13.605698066,
        "J": 6.24150934e18,
        "kJ": 6.24150934e21,
    },
    "length": {
        "ang": 1,
        "m": 1e10,
        "cm": 1e8,
        "pm": 1e-2,
        "bohr": 0.5291772083,
    },
    "mass": {
        "kg": 1,
        "g": 1e-3,
        "amu": 1.660538921e-27,
    },
    "temperature": {
        "K": 1,
    },
    "time": {
        "s": 1,
        "min": 60,
        "h": 3600,
    },
    "charge": {
        "C": 1,
        "e": 1.602176565e-19,
    },
    "angle": {
        "rad": 180,
        "deg": math.pi
    },
    "amount": {
        "atom": 1,
        "mol": 6.02214129e23
    },
    "force": {
        "N": 1
    }
}

DERIVED_UNITS = {
    "J": {"kg": 1, "m": 2, "s": -2},
    "N": {"kg": 1, "m": 1, "s": -2}
}


# Mapping unit name --> unit type (unit names must be unique).
_UNAME2UTYPE = {}
for utype, d in SUPPORTED_UNITS.items():
    assert not set(d.keys()).intersection(_UNAME2UTYPE.keys())
    _UNAME2UTYPE.update({uname: utype for uname in d})
del utype, d


# TODO
# One can use unit_type_from_unit_name to reduce the number of arguments
# passed to the decorator.
def unit_type_from_unit_name(uname):
    """Return the unit type from the unit name."""
    return _UNAME2UTYPE[uname]


class Unit(collections.Mapping):
    """
    Represents a unit, e.g., "m" for meters, etc. Supports compound units.
    Only integer powers are supported for units.
    """
    def __init__(self, unit_def):
        """
        Constructs a unit.

        Args:
            unit_def:
                A definition for the unit. Either a mapping of unit to
                powers, e.g., {"m": 2, "s": -1} represents "m^2 s^-1",
                or simply as a string "kg m^2 s^-1".
        """

        if isinstance(unit_def, basestring):
            unit = collections.defaultdict(int)
            for m in re.finditer("([A-Za-z]+)\s*\^*\s*([\-0-9]*)", unit_def):
                p = m.group(2)
                p = 1 if not p else int(p)
                unit[m.group(1)] += p
        else:
            unit = dict(unit_def)

        #Convert to base units
        base_units = collections.defaultdict(int)
        for u, p in unit.items():
            if u in DERIVED_UNITS:
                for k, v in DERIVED_UNITS[u].items():
                    base_units[k] += v * p
            else:
                base_units[u] += p

        def check_mappings(u):
            for k, v in DERIVED_UNITS.items():
                if u == v:
                    self._unit = {k: 1}
                    return
            self._unit = u

        check_mappings(base_units)

    def __mul__(self, other):
        new_units = collections.defaultdict(int)
        for k, v in self.items():
            new_units[k] += v
        for k, v in other.items():
            new_units[k] += v
        return Unit(new_units)

    def __rmul__(self, other):
        return self.__mult__(other)

    def __div__(self, other):
        new_units = collections.defaultdict(int)
        for k, v in self.items():
            new_units[k] += v
        for k, v in other.items():
            new_units[k] -= v
        return Unit(new_units)

    def __truediv__(self, other):
        return self.__div__(other)

    def __pow__(self, i):
        return Unit({k: v * i for k, v in self.items()})

    def __iter__(self):
        return self._unit.__iter__()

    def __getitem__(self, i):
        return self._unit[i]

    def __len__(self):
        return len(self._unit)

    def __repr__(self):
        sorted_keys = sorted(self._unit.keys(),
                             key=lambda k: (-self._unit[k], k))
        return " ".join(["{}^{}".format(k, self._unit[k])
                         if self._unit[k] != 1 else k
                         for k in sorted_keys if self._unit[k] != 0])

    def __str__(self):
        return self.__repr__()

    @property
    def as_base_units_dict(self):
        b = collections.defaultdict(int)
        for k, v in self.items():
            if k in DERIVED_UNITS:
                for k2, v2 in DERIVED_UNITS[k].items():
                    b[k2] += v2
            else:
                b[k] += v
        return b

    def get_conversion_factor(self, new_unit):
        """
        Returns a conversion factor between this unit and a new unit.
        Compound units are supported, but must have the same powers in each
        unit type.

        Args:
            new_unit:
                The new unit.
        """
        units_new = sorted(((k, v) for k, v in Unit(new_unit).items()),
                           key=lambda d: _UNAME2UTYPE[d[0]])
        units_old = sorted(((k, v) for k, v in self.items()),
                           key=lambda d: _UNAME2UTYPE[d[0]])
        factor = 1
        for uo, un in zip(units_old, units_new):
            if uo[1] != un[1]:
                raise UnitError("Units are not compatible!")
            c = SUPPORTED_UNITS[_UNAME2UTYPE[uo[0]]]
            factor *= (c[uo[0]] / c[un[0]]) ** uo[1]
        return factor


class FloatWithUnit(float):
    """
    Subclasses float to attach a unit type. Typically, you should use the
    pre-defined unit type subclasses such as Energy, Length, etc. instead of
    using FloatWithUnit directly.

    Supports conversion, addition and subtraction of the same unit type. E.g.,
    1 m + 20 cm will be automatically converted to 1.2 m (units follow the
    leftmost quantity).

    >>> e = Energy(1.1, "Ha")
    >>> a = Energy(1.1, "Ha")
    >>> b = Energy(3, "eV")
    >>> c = a + b
    >>> print c
    1.2102479761938871 Ha
    >>> c.to("eV")
    32.932522246000005 eV
    """

    def __new__(cls, val, unit, unit_type=None):
        return float.__new__(cls, val)

    def __init__(self, val, unit, unit_type=None):
        self._unit_type = unit_type
        if unit_type is not None and str(unit) not in SUPPORTED_UNITS[unit_type]:
            raise UnitError(
                "{} is not a supported unit for {}".format(unit, unit_type))
        self._unit = Unit(unit)
        super(FloatWithUnit, self).__init__(val)

    def __repr__(self):
        s = super(FloatWithUnit, self).__repr__()
        return "{} {}".format(s, self._unit)

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if not hasattr(other, "unit_type"):
            return super(FloatWithUnit, self).__add__(other)
        if other.unit_type != self._unit_type:
            raise UnitError("Adding different types of units is not allowed")
        val = other
        if other.unit != self._unit:
            val = other.to(self._unit)
        return FloatWithUnit(float(self) + val, unit_type=self._unit_type,
                             unit=self._unit)

    def __sub__(self, other):
        if not hasattr(other, "unit_type"):
            return super(FloatWithUnit, self).__sub__(other)
        if other.unit_type != self._unit_type:
            raise UnitError("Subtracting different units is not allowed")
        val = other
        if other.unit != self._unit:
            val = other.to(self._unit)
        return FloatWithUnit(float(self) - val, unit_type=self._unit_type,
                             unit=self._unit)

    def __mul__(self, other):
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(float(self) * other,
                                 unit_type=self._unit_type,
                                 unit=self._unit)
        return FloatWithUnit(float(self) * other, unit_type=None,
                             unit=self._unit * other._unit)

    def __rmul__(self, other):
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(float(self) * other,
                                 unit_type=self._unit_type,
                                 unit=self._unit)
        return FloatWithUnit(float(self) * other, unit_type=None,
                             unit=self._unit * other._unit)
    def __pow__(self, i):
        return FloatWithUnit(float(self) ** i, unit_type=None,
                             unit=self._unit ** i)

    def __div__(self, other):
        val = super(FloatWithUnit, self).__div__(other)
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(val, unit_type=self._unit_type,
                                 unit=self._unit)
        return FloatWithUnit(val, unit_type=None,
                             unit=self._unit / other._unit)

    def __truediv__(self, other):
        val = super(FloatWithUnit, self).__truediv__(other)
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(val, unit_type=self._unit_type,
                                 unit=self._unit)
        return FloatWithUnit(val, unit_type=None,
                             unit=self._unit / other._unit)

    def __neg__(self):
        return FloatWithUnit(super(FloatWithUnit, self).__neg__(),
                             unit_type=self._unit_type,
                             unit=self._unit)

    @property
    def unit_type(self):
        return self._unit_type

    @property
    def unit(self):
        return self._unit

    def to(self, new_unit):
        """
        Conversion to a new_unit. Right now, only supports 1 to 1 mapping of
        units of each type.

        Args:
            new_unit:
                New unit type.

        Returns:
            A FloatWithUnit object in the new units.

        Example usage:
        >>> e = Energy(1.1, "eV")
        >>> e = Energy(1.1, "Ha")
        >>> e.to("eV")
        29.932522246000005 eV
        """
        return FloatWithUnit(
            self * self.unit.get_conversion_factor(new_unit),
            unit_type=self._unit_type,
            unit=new_unit)

    @property
    def supported_units(self):
        """
        Supported units for specific unit type.
        """
        return SUPPORTED_UNITS[self._unit_type]


class ArrayWithUnit(np.ndarray):
    """
    Subclasses `numpy.ndarray` to attach a unit type. Typically, you should
    use the pre-defined unit type subclasses such as EnergyArray,
    LengthArray, etc. instead of using ArrayWithFloatWithUnit directly.

    Supports conversion, addition and subtraction of the same unit type. E.g.,
    1 m + 20 cm will be automatically converted to 1.2 m (units follow the
    leftmost quantity).

    >>> a = EnergyArray([1, 2], "Ha")
    >>> b = EnergyArray([1, 2], "eV")
    >>> c = a + b
    >>> print c
    [ 1.03674933  2.07349865] Ha
    >>> c.to("eV")
    array([ 28.21138386,  56.42276772]) eV
    """
    def __new__(cls, input_array, unit, unit_type=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj._unit = Unit(unit)
        obj._unit_type = unit_type
        return obj

    def __array_finalize__(self, obj):
        """
        See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
        comments.
        """
        if obj is None:
            return
        self._unit = getattr(obj, "_unit", None)
        self._unit_type = getattr(obj, "_unit_type", None)

    #TODO abstract base class property?
    @property
    def unit_type(self):
        return self._unit_type

    #TODO abstract base class property?
    @property
    def unit(self):
        return self._unit

    def __repr__(self):
        return "{} {}".format(np.array(self).__repr__(), self.unit)

    def __str__(self):
        return "{} {}".format(np.array(self).__str__(), self.unit)

    def __add__(self, other):
        if hasattr(other, "unit_type"):
            if other.unit_type != self.unit_type:
                raise UnitError("Adding different types of units is"
                                 " not allowed")

            if other.unit != self.unit:
                other = other.to(self.unit)

        return self.__class__(np.array(self) + np.array(other),
                              unit_type=self.unit_type, unit=self.unit)

    def __sub__(self, other):
        if hasattr(other, "unit_type"):
            if other.unit_type != self.unit_type:
                raise UnitError("Subtracting different units is not allowed")

            if other.unit != self.unit:
                other = other.to(self.unit)

        return self.__class__(np.array(self) - np.array(other),
                              unit_type=self.unit_type, unit=self.unit)

    def __mul__(self, other):
        # FIXME
        # Here we have the most important difference between FloatWithUnit and
        # ArrayWithFloatWithUnit:
        # If other does not have units, I return an object with the same units
        # as self.
        # if other *has* units, I return an object *without* units since
        # taking into account all the possible derived quantities would be
        # too difficult.
        # Moreover Energy(1.0) * Time(1.0, "s") returns 1.0 Ha that is a
        # bit misleading.
        # Same protocol for __div__
        if not hasattr(other, "unit_type"):
            return self.__class__(np.array(self).__mul__(np.array(other)),
                                  unit_type=self._unit_type, unit=self._unit)
        else:
            # Cannot use super since it returns an instance of self.__class__
            # while here we want a bare numpy array.
            return np.array(self).__mul__(np.array(other))

    def __rmul__(self, other):
        if not hasattr(other, "unit_type"):
            return self.__class__(np.array(self).__rmul__(np.array(other)),
                                  unit_type=self._unit_type, unit=self._unit)
        else:
            return np.array(self).__rmul__(np.array(other))

    def __div__(self, other):
        if not hasattr(other, "unit_type"):
            return self.__class__(np.array(self).__div__(np.array(other)),
                                  unit_type=self._unit_type, unit=self._unit)
        else:
            return np.array(self).__div__(np.array(other))

    def __truediv__(self, other):
        if not hasattr(other, "unit_type"):
            return self.__class__(np.array(self).__truediv__(np.array(other)),
                                  unit_type=self._unit_type, unit=self._unit)
        else:
            return np.array(self).__truediv__(np.array(other))

    def __neg__(self):
        return self.__class__(np.array(self).__neg__(),
                              unit_type=self.unit_type, unit=self.unit)

    def to(self, new_unit):
        """
        Conversion to a new_unit.

        Args:
            new_unit:
                New unit type.

        Returns:
            A ArrayWithFloatWithUnit object in the new units.

        Example usage:
        >>> e = EnergyArray([1, 1.1], "Ha")
        >>> e.to("eV")
        array([ 27.21138386,  29.93252225]) eV
        """
        return self.__class__(
            np.array(self) * self.unit.get_conversion_factor(new_unit),
            unit_type=self.unit_type, unit=new_unit)

    #TODO abstract base class property?
    @property
    def supported_units(self):
        """
        Supported units for specific unit type.
        """
        return SUPPORTED_UNITS[self.unit_type]

    #TODO abstract base class method?
    def conversions(self):
        """
        Returns a string showing the available conversions.
        Useful tool in interactive mode.
        """
        return "\n".join(str(self.to(unit)) for unit in self.supported_units)


Energy = partial(FloatWithUnit, unit_type="energy")
EnergyArray = partial(ArrayWithUnit, unit_type="energy")

Length = partial(FloatWithUnit, unit_type="length")
LengthArray = partial(ArrayWithUnit, unit_type="length")

Mass = partial(FloatWithUnit, unit_type="mass")
MassArray = partial(ArrayWithUnit, unit_type="mass")

Temp = partial(FloatWithUnit, unit_type="temperature")
TempArray = partial(ArrayWithUnit, unit_type="temperature")

Time = partial(FloatWithUnit, unit_type="time")
TimeArray = partial(ArrayWithUnit, unit_type="time")

Charge = partial(FloatWithUnit, unit_type="charge")
ChargeArray = partial(ArrayWithUnit, unit_type="charge")

Angle = partial(FloatWithUnit, unit_type="angle")
AngleArray = partial(ArrayWithUnit, unit_type="angle")


def obj_with_unit(obj, unit):
    """
    Returns a `FloatWithUnit` instance if obj is scalar, a dictionary of
    objects with units if obj is a dict, else an instance of
    `ArrayWithFloatWithUnit`.

    Args:
        unit:
            Specific units (eV, Ha, m, ang, etc.).
    """
    unit_type = unit_type_from_unit_name(unit)

    if isinstance(obj, numbers.Number):
        return FloatWithUnit(obj, unit=unit, unit_type=unit_type)
    elif isinstance(obj, collections.Mapping):
        return {k: obj_with_unit(v, unit) for k,v in obj.items()}
    else:
        return ArrayWithUnit(obj, unit=unit, unit_type=unit_type)


def unitized(unit):
    """
    Useful decorator to assign units to the output of a function. For
    sequences, all values in the sequences are assigned the same unit. It
    works with Python sequences only. The creation of numpy arrays loses all
    unit information. For mapping types, the values are assigned units.

    Args:
        units:
            Specific units (eV, Ha, m, ang, etc.).
    """
    def wrap(f):
        def wrapped_f(*args, **kwargs):
            val = f(*args, **kwargs)
            unit_type = unit_type_from_unit_name(unit)
            if isinstance(val, collections.Sequence):
                # TODO: why don't we return a ArrayWithFloatWithUnit?
                # This complicated way is to ensure the sequence type is
                # preserved (list or tuple).
                return val.__class__([FloatWithUnit(i, unit_type=unit_type,
                                           unit=unit) for i in val])
            elif isinstance(val, collections.Mapping):
                for k, v in val.items():
                    val[k] = FloatWithUnit(v, unit_type=unit_type, unit=unit)
            elif isinstance(val, numbers.Number):
                return FloatWithUnit(val, unit_type=unit_type, unit=unit)
            return val
        return wrapped_f
    return wrap


class UnitError(BaseException):
    """
    Exception class for unit errors.
    """
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
