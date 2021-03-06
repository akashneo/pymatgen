#!/usr/bin/env python
from __future__ import division, print_function

import unittest
import os.path


from pymatgen.core.structure import Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.io.abinitio.abiobjects import *

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files')


def cif_paths():
    cifpaths = []
    print(test_dir)
    for fname in os.listdir(test_dir):
        fname = os.path.join(test_dir, fname)
        if os.path.isfile(fname) and fname.endswith(".cif"):
            cifpaths.append(fname)

    assert cifpaths
    return cifpaths

##########################################################################################

class SpinModeTest(unittest.TestCase):

    def test_base(self):
        polarized = SpinMode.asspinmode("polarized")
        other_polarized = SpinMode.asspinmode("polarized")
        unpolarized = SpinMode.asspinmode("unpolarized")

        polarized.to_abivars()

        self.assertTrue(polarized is other_polarized)
        self.assertTrue(polarized == other_polarized)
        self.assertTrue(polarized != unpolarized)

##########################################################################################

class SmearingTest(unittest.TestCase):
    def test_base(self):
        fd1ev = Smearing.assmearing("fermi_dirac:1 eV")
        print(fd1ev)
        fd1ev.to_abivars()

        self.assertTrue(fd1ev)

        same_fd = Smearing.assmearing("fermi_dirac:"+ str(1.0/Ha_to_eV))

        self.assertTrue(same_fd == fd1ev)

        nosmear = Smearing.nosmearing()

        self.assertFalse(nosmear)
        self.assertTrue(nosmear != fd1ev)
        new_fd1ev = Smearing.from_dict(fd1ev.to_dict)
        self.assertTrue(new_fd1ev == fd1ev)

##########################################################################################

class ElectronsAlgorithmTest(unittest.TestCase):
    def test_base(self):
        algo = ElectronsAlgorithm(nstep=70)
        print(algo.to_abivars())

##########################################################################################

class ElectronsTest(unittest.TestCase):
    def test_base(self):
        default_electrons = Electrons()
        self.assertTrue(default_electrons.nsppol==2)
        self.assertTrue(default_electrons.nspinor==1)
        self.assertTrue(default_electrons.nspden==2)

        print(default_electrons.to_abivars())

        #new = Electron.from_dict(default_electrons.to_dict())

##########################################################################################

class AbiStructureTest(unittest.TestCase):

    def setUp(self):
        self.cif_paths = cif_paths()

    def test_asabistructure(self):
        for cif_path in self.cif_paths:
            print("about to init abistructure from %s " % cif_path)
            st = asabistructure(cif_path)
            self.assertTrue(st is asabistructure(st))
            self.assertTrue(isinstance(st, Structure))

            # FIXME
            if not st.is_ordered:
                print("Unordered structures are not supported")
                continue

            print(st.to_abivars())

##########################################################################################

#class KSamplingTest(unittest.TestCase):

##########################################################################################

#class RelaxationTest(unittest.TestCase):

##########################################################################################

class PPModelTest(unittest.TestCase):

    def test_base(self):
        godby = PPModel.asppmodel("godby:12 eV")
        print(godby)
        print(repr(godby))
        godby.to_abivars()
        self.assertTrue(godby)

        same_godby = PPModel.asppmodel("godby:"+ str(12.0/Ha_to_eV))
        self.assertTrue(same_godby == godby)

        noppm = PPModel.noppmodel()

        self.assertFalse(noppm)
        self.assertTrue(noppm != godby)
        new_godby = PPModel.from_dict(godby.to_dict)
        self.assertTrue(new_godby == godby)

##########################################################################################

if __name__ == '__main__':
    unittest.main()
