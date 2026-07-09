// TIDES nanobind bindings — T10.1
//
// Bridges the C++20 core to Python via nanobind. When built, this produces
// the `_native` extension module imported by tides.core. When not built,
// the Python reference implementation is used.
//
// Build: see CMakeLists.txt target `tides_python_bindings`.
// Coding standard: C++ exceptions never cross the boundary; all bindings
// return Status/result objects to Python.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/dict.h>
#include <nanobind/stl/function.h>

#include "common/status.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "forces/analytic_forces.hpp"
#include "dynamics/xlbomd/xlbomd.hpp"
#include "dynamics/optimizers/optimizers.hpp"
#include "solvers/broker.hpp"

namespace nb = nanobind;
using namespace tides;

NB_MODULE(_native, m) {
    m.doc() = "TIDES native C++ backend (nanobind bindings)";

    // Status / StatusCode
    nb::enum_<StatusCode>(m, "StatusCode")
        .value("OK", StatusCode::kOk)
        .value("INVALID_ARGUMENT", StatusCode::kInvalidArgument)
        .value("OUT_OF_RANGE", StatusCode::kOutOfRange)
        .value("IO_ERROR", StatusCode::kIoError)
        .value("CORRUPT_DATA", StatusCode::kCorruptData)
        .value("UNIMPLEMENTED", StatusCode::kUnimplemented);

    nb::class_<Status>(m, "CppStatus")
        .def_prop_ro("ok", &Status::ok)
        .def_prop_ro("code", &Status::code)
        .def_prop_ro("message", &Status::message)
        .def_static("ok", &Status::Ok)
        .def_static("invalid_argument", &Status::InvalidArgument)
        .def_static("io_error", &Status::IoError)
        .def_static("unimplemented", &Status::Unimplemented);

    // SCFResult
    nb::class_<scf::SCFResult>(m, "CppSCFResult")
        .def_ro("energy", &scf::SCFResult::energy)
        .def_ro("P", &scf::SCFResult::P)
        .def_ro("eigenvalues", &scf::SCFResult::eigenvalues)
        .def_ro("eigenvectors", &scf::SCFResult::eigenvectors)
        .def_ro("n_iterations", &scf::SCFResult::n_iterations)
        .def_ro("converged", &scf::SCFResult::converged)
        .def_ro("energy_history", &scf::SCFResult::energy_history);

    // EnergyComponents
    nb::class_<scf::EnergyComponents>(m, "CppEnergyComponents")
        .def_ro("E_kin", &scf::EnergyComponents::E_kin)
        .def_ro("E_ne", &scf::EnergyComponents::E_ne)
        .def_ro("E_H", &scf::EnergyComponents::E_H)
        .def_ro("E_xc", &scf::EnergyComponents::E_xc)
        .def_ro("E_ion", &scf::EnergyComponents::E_ion)
        .def_ro("E_total", &scf::EnergyComponents::E_total);

    // SolverBroker
    nb::class_<solvers::SolverBroker>(m, "SolverBroker")
        .def_static("dispatch", &solvers::SolverBroker::Dispatch)
        .def_static("generate_calib_table", &solvers::SolverBroker::GenerateCalibTable);

    // SolverRegime
    nb::enum_<solvers::SolverRegime>(m, "SolverRegime")
        .value("R0_BatchDense", solvers::SolverRegime::kR0_BatchDense)
        .value("R1_ChFSI", solvers::SolverRegime::kR1_ChFSI)
        .value("R2_SP2", solvers::SolverRegime::kR2_SP2)
        .value("R3_FOE_SQ", solvers::SolverRegime::kR3_FOE_SQ);

    // ForceResult
    nb::class_<forces::ForceResult>(m, "ForceResult")
        .def_ro("forces", &forces::ForceResult::forces)
        .def_ro("fd_validated", &forces::ForceResult::fd_validated)
        .def_ro("max_fd_error", &forces::ForceResult::max_fd_error);

    // AnalyticForces
    nb::class_<forces::AnalyticForces>(m, "AnalyticForces")
        .def_static("fd5_force", &forces::AnalyticForces::FD5Force)
        .def_static("hellmann_feynman", &forces::AnalyticForces::HellmannFeynman)
        .def_static("validate", &forces::AnalyticForces::Validate);

    // XLBOMDResult
    nb::class_<dynamics::XLBOMDResult>(m, "XLBOMDResult")
        .def_ro("n_steps", &dynamics::XLBOMDResult::n_steps)
        .def_ro("total_drift", &dynamics::XLBOMDResult::total_drift)
        .def_ro("avg_solves_per_step", &dynamics::XLBOMDResult::avg_solves_per_step)
        .def_ro("energy_history", &dynamics::XLBOMDResult::energy_history);

    // XLBOMD
    nb::class_<dynamics::XLBOMD>(m, "XLBOMD")
        .def_static("run", &dynamics::XLBOMD::Run,
            nb::arg("init_R"), nb::arg("masses"), nb::arg("dt"),
            nb::arg("n_steps"), nb::arg("force_fn"), nb::arg("energy_fn"),
            nb::arg("density_fn"), nb::arg("thermostat") = 0,
            nb::arg("kT") = 0.0);

    // SCFDriver
    nb::class_<scf::SCFDriver>(m, "SCFDriver")
        .def_static("run", &scf::SCFDriver::Run,
            nb::arg("n"), nb::arg("n_occ"), nb::arg("S"),
            nb::arg("build_H"), nb::arg("energy_fn"),
            nb::arg("P_init") = std::vector<double>{},
            nb::arg("max_iter") = 100, nb::arg("tol") = 1e-10,
            nb::arg("mixing") = 1, nb::arg("alpha") = 0.3);

    // EnergyAssembly
    nb::class_<scf::EnergyAssembly>(m, "EnergyAssembly")
        .def_static("compute", &scf::EnergyAssembly::Compute)
        .def_static("ewald_ion_ion", &scf::EnergyAssembly::EwaldIonIon);

    // BrokerInput
    nb::class_<solvers::BrokerInput>(m, "BrokerInput")
        .def_rw("n_atoms", &solvers::BrokerInput::n_atoms)
        .def_rw("n_basis", &solvers::BrokerInput::n_basis)
        .def_rw("bc_type", &solvers::BrokerInput::bc_type)
        .def_rw("gap_estimate", &solvers::BrokerInput::gap_estimate)
        .def_rw("electronic_temp", &solvers::BrokerInput::electronic_temp)
        .def_rw("available_vram_mb", &solvers::BrokerInput::available_vram_mb)
        .def_rw("user_override", &solvers::BrokerInput::user_override)
        .def_rw("forced_regime", &solvers::BrokerInput::forced_regime);

    // CalibEntry
    nb::class_<solvers::CalibEntry>(m, "CalibEntry")
        .def_rw("regime", &solvers::CalibEntry::regime)
        .def_rw("n_atoms_lo", &solvers::CalibEntry::n_atoms_lo)
        .def_rw("n_atoms_hi", &solvers::CalibEntry::n_atoms_hi)
        .def_rw("time_per_step_ms", &solvers::CalibEntry::time_per_step_ms)
        .def_rw("vram_mb", &solvers::CalibEntry::vram_mb)
        .def_rw("available", &solvers::CalibEntry::available);

    m.attr("__version__") = "0.1.0-alpha";
}
