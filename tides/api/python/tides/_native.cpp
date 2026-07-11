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
#include <nanobind/stl/function.h>

#include "common/status.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "scf/molecule_driver.hpp"
#include "scf/nao_driver.hpp"
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
        .def_prop_ro("message", &Status::message);
    m.def("status_ok", []() { return Status::Ok(); });
    m.def("status_invalid_argument", [](const std::string& msg) { return Status::InvalidArgument(msg); });
    m.def("status_io_error", [](const std::string& msg) { return Status::IoError(msg); });
    m.def("status_unimplemented", [](const std::string& msg) { return Status::Unimplemented(msg); });

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
        .def_static("dispatch", [](const solvers::BrokerInput& input,
                                   const std::vector<solvers::CalibEntry>& calib) {
            std::string reason;
            auto regime = solvers::SolverBroker::Dispatch(input, calib, reason);
            return nb::make_tuple(regime, reason);
        }, nb::arg("input"), nb::arg("calib"))
        .def_static("generate_calib_table", []() {
            return solvers::SolverBroker::GenerateCalibTable();
        });

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
        .def_static("fd5_force", [](const std::function<double(const std::vector<double>&)>& energy_fn,
                                    std::vector<double> positions, std::size_t atom_idx,
                                    int component, double h = 0.001) {
            return forces::AnalyticForces::FD5Force(energy_fn, positions, atom_idx, component, h);
        }, nb::arg("energy_fn"), nb::arg("positions"), nb::arg("atom_idx"),
           nb::arg("component"), nb::arg("h") = 0.001)
        .def_static("hellmann_feynman", [](const std::vector<double>& P,
                                            const std::vector<std::vector<double>>& dH_dR,
                                            std::size_t n, std::size_t n_atoms) {
            return forces::AnalyticForces::HellmannFeynman(P, dH_dR, n, n_atoms);
        }, nb::arg("P"), nb::arg("dH_dR"), nb::arg("n"), nb::arg("n_atoms"))
        .def_static("validate", [](const std::vector<double>& analytic_forces,
                                    const std::function<double(const std::vector<double>&)>& energy_fn,
                                    const std::vector<double>& positions, std::size_t n_atoms,
                                    double h = 0.001, double tol = 1e-4) {
            return forces::AnalyticForces::Validate(analytic_forces, energy_fn, positions, n_atoms, h, tol);
        }, nb::arg("analytic_forces"), nb::arg("energy_fn"), nb::arg("positions"),
           nb::arg("n_atoms"), nb::arg("h") = 0.001, nb::arg("tol") = 1e-4);

    // XLBOMDResult
    nb::class_<dynamics::XLBOMDResult>(m, "XLBOMDResult")
        .def_ro("n_steps", &dynamics::XLBOMDResult::n_steps)
        .def_ro("total_drift", &dynamics::XLBOMDResult::total_drift)
        .def_ro("avg_solves_per_step", &dynamics::XLBOMDResult::avg_solves_per_step)
        .def_ro("energy_history", &dynamics::XLBOMDResult::energy_history);

    // XLBOMD
    nb::class_<dynamics::XLBOMD>(m, "XLBOMD")
        .def_static("run", [](const std::vector<double>& init_R, const std::vector<double>& masses,
                              double dt, int n_steps,
                              const std::function<std::vector<double>(const std::vector<double>&)>& force_fn,
                              const std::function<double(const std::vector<double>&)>& energy_fn,
                              const std::function<std::vector<double>(const std::vector<double>&)>& density_fn,
                              int thermostat = 0, double kT = 0.0) {
            return dynamics::XLBOMD::Run(init_R, masses, dt, n_steps, force_fn, energy_fn, density_fn, thermostat, kT);
        }, nb::arg("init_R"), nb::arg("masses"), nb::arg("dt"),
           nb::arg("n_steps"), nb::arg("force_fn"), nb::arg("energy_fn"),
           nb::arg("density_fn"), nb::arg("thermostat") = 0,
           nb::arg("kT") = 0.0);

    // SCFDriver
    // AUDIT B5/B7: energy_fn now receives eigenvalues from the SCF loop.
    nb::class_<scf::SCFDriver>(m, "SCFDriver")
        .def_static("run", [](std::size_t n, std::size_t n_occ, const std::vector<double>& S,
                              const std::function<std::vector<double>(const std::vector<double>&)>& build_H,
                              const std::function<double(const std::vector<double>&, const std::vector<double>&)>& energy_fn,
                              const std::vector<double>& P_init = {},
                              int max_iter = 100, double tol = 1e-10,
                              int mixing = 1, double alpha = 0.3) {
            return scf::SCFDriver::Run(n, n_occ, S, build_H, energy_fn, P_init, max_iter, tol, mixing, alpha);
        }, nb::arg("n"), nb::arg("n_occ"), nb::arg("S"),
           nb::arg("build_H"), nb::arg("energy_fn"),
           nb::arg("P_init") = std::vector<double>{},
           nb::arg("max_iter") = 100, nb::arg("tol") = 1e-10,
           nb::arg("mixing") = 1, nb::arg("alpha") = 0.3);

    // EnergyAssembly
    nb::class_<scf::EnergyAssembly>(m, "EnergyAssembly")
        .def_static("compute", [](double E_kin, const std::vector<double>& P,
                                  const std::vector<double>& H, const std::vector<double>& Vne,
                                  const std::vector<double>& Vh, const std::vector<double>& Vxc,
                                  const std::vector<double>& S, std::size_t n, double E_ion_ion = 0.0) {
            return scf::EnergyAssembly::Compute(E_kin, P, H, Vne, Vh, Vxc, S, n, E_ion_ion);
        }, nb::arg("E_kin"), nb::arg("P"), nb::arg("H"), nb::arg("Vne"),
           nb::arg("Vh"), nb::arg("Vxc"), nb::arg("S"), nb::arg("n"),
           nb::arg("E_ion_ion") = 0.0)
        .def_static("ewald_ion_ion", [](const std::vector<double>& positions,
                                         const std::vector<double>& charges,
                                         bool periodic = false, double alpha = 0.0) {
            return scf::EnergyAssembly::EwaldIonIon(positions, charges, periodic, alpha);
        }, nb::arg("positions"), nb::arg("charges"),
           nb::arg("periodic") = false, nb::arg("alpha") = 0.0);

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

    // AUDIT C7: MoleculeDriver bindings — the real GTO-based SCF engine.
    // This replaces the model Hamiltonian stub in Python.
    // EnergyComponents already bound above as CppEnergyComponents.

    // PipelineTimings: per-component profiling data (Audit P3).
    nb::class_<scf::PipelineTimings>(m, "PipelineTimings")
        .def_ro("rho_build_ms", &scf::PipelineTimings::rho_build_ms)
        .def_ro("xc_eval_ms", &scf::PipelineTimings::xc_eval_ms)
        .def_ro("poisson_ms", &scf::PipelineTimings::poisson_ms)
        .def_ro("vmat_build_ms", &scf::PipelineTimings::vmat_build_ms)
        .def_ro("eigensolve_ms", &scf::PipelineTimings::eigensolve_ms)
        .def_ro("scf_total_ms", &scf::PipelineTimings::scf_total_ms)
        .def_ro("n_iterations", &scf::PipelineTimings::n_iterations)
        .def_ro("used_gpu_xc", &scf::PipelineTimings::used_gpu_xc)
        .def_ro("used_grid_hartree", &scf::PipelineTimings::used_grid_hartree)
        .def_ro("xc_functional", &scf::PipelineTimings::xc_functional);

    nb::class_<scf::MoleculeDriverResult>(m, "MoleculeDriverResult")
        .def_rw("scf", &scf::MoleculeDriverResult::scf)
        .def_rw("energy", &scf::MoleculeDriverResult::energy)
        .def_rw("n_basis", &scf::MoleculeDriverResult::n_basis)
        .def_rw("n_electrons", &scf::MoleculeDriverResult::n_electrons)
        .def_rw("n_atoms", &scf::MoleculeDriverResult::n_atoms)
        .def_rw("grid_h", &scf::MoleculeDriverResult::grid_h)
        .def_rw("wall_time_ms", &scf::MoleculeDriverResult::wall_time_ms)
        .def_rw("forces", &scf::MoleculeDriverResult::forces)
        .def_rw("timings", &scf::MoleculeDriverResult::timings)
        .def_prop_ro("grid_n", [](const scf::MoleculeDriverResult& r) {
            return std::vector<std::size_t>{r.grid_n[0], r.grid_n[1], r.grid_n[2]};
        });

    nb::class_<scf::GTOMolecule>(m, "GTOMolecule")
        .def_rw("atomic_numbers", &scf::GTOMolecule::atomic_numbers)
        .def_rw("positions", &scf::GTOMolecule::positions)
        .def_rw("n_basis", &scf::GTOMolecule::n_basis);

    nb::class_<scf::MoleculeDriver>(m, "MoleculeDriver")
        .def_static("build_molecule", [](const std::vector<int>& atomic_numbers,
                                         const std::vector<double>& positions) {
            return scf::MoleculeDriver::BuildMolecule(atomic_numbers, positions);
        }, nb::arg("atomic_numbers"), nb::arg("positions"))
        .def_static("run", [](const scf::GTOMolecule& mol,
                              double grid_h, double grid_margin,
                              int max_iter, double tol,
                              bool use_grid_hartree,
                              const std::string& xc_functional) {
            // Map string to XcSpec.
            grid::xc::HostXcSpec spec{};
            if (xc_functional == "pbe" || xc_functional == "PBE") {
                spec.id = grid::xc::XcFunctionalId::kPbe;
                spec.family = grid::xc::XcFamily::kGga;
            } else if (xc_functional == "pbesol" || xc_functional == "PBEsol") {
                spec.id = grid::xc::XcFunctionalId::kPbesol;
                spec.family = grid::xc::XcFamily::kGga;
            } else if (xc_functional == "revpbe" || xc_functional == "revPBE") {
                spec.id = grid::xc::XcFunctionalId::kRevPbe;
                spec.family = grid::xc::XcFamily::kGga;
            } else {
                // Default: LDA-PW92.
                spec.id = grid::xc::XcFunctionalId::kLdaPw92;
                spec.family = grid::xc::XcFamily::kLda;
            }
            return scf::MoleculeDriver::Run(mol, grid_h, grid_margin, max_iter, tol,
                                             use_grid_hartree, spec, false);
        }, nb::arg("mol"),
           nb::arg("grid_h") = 0.3, nb::arg("grid_margin") = 4.0,
           nb::arg("max_iter") = 100, nb::arg("tol") = 1e-8,
           nb::arg("use_grid_hartree") = false,
           nb::arg("xc_functional") = std::string("lda"))
        .def_static("compute_forces", [](const scf::GTOMolecule& mol,
                                         const scf::SCFResult& scf_result,
                                         const scf::EnergyComponents& energy) {
            return scf::MoleculeDriver::ComputeForces(mol, scf_result, energy);
        }, nb::arg("mol"), nb::arg("scf_result"), nb::arg("energy"));

    // NaoDriver — the NAO-based SCF engine (product pipeline).
    nb::class_<scf::NaoDriverResult>(m, "NaoDriverResult")
        .def_rw("scf", &scf::NaoDriverResult::scf)
        .def_rw("energy", &scf::NaoDriverResult::energy)
        .def_rw("n_basis", &scf::NaoDriverResult::n_basis)
        .def_rw("n_electrons", &scf::NaoDriverResult::n_electrons)
        .def_rw("n_atoms", &scf::NaoDriverResult::n_atoms)
        .def_rw("grid_h", &scf::NaoDriverResult::grid_h)
        .def_rw("wall_time_ms", &scf::NaoDriverResult::wall_time_ms)
        .def_rw("basis_info", &scf::NaoDriverResult::basis_info)
        .def_prop_ro("grid_n", [](const scf::NaoDriverResult& r) {
            return std::vector<std::size_t>{r.grid_n[0], r.grid_n[1], r.grid_n[2]};
        });

    nb::class_<scf::NaoDriver>(m, "NaoDriver")
        .def_static("run", [](const std::vector<int>& atomic_numbers,
                              const std::vector<double>& positions,
                              double grid_h, double grid_margin,
                              int max_iter, double tol) {
            return scf::NaoDriver::Run(atomic_numbers, positions,
                                        grid_h, grid_margin, max_iter, tol);
        }, nb::arg("atomic_numbers"), nb::arg("positions"),
           nb::arg("grid_h") = 0.3, nb::arg("grid_margin") = 4.0,
           nb::arg("max_iter") = 100, nb::arg("tol") = 1e-8)
        .def_static("compute_forces", [](const std::vector<int>& atomic_numbers,
                                          const std::vector<double>& positions,
                                          double grid_h, double grid_margin,
                                          int max_iter, double tol,
                                          double h) {
            return scf::NaoDriver::ComputeForces(atomic_numbers, positions,
                                                  grid_h, grid_margin,
                                                  max_iter, tol, h);
        }, nb::arg("atomic_numbers"), nb::arg("positions"),
           nb::arg("grid_h") = 0.3, nb::arg("grid_margin") = 4.0,
           nb::arg("max_iter") = 50, nb::arg("tol") = 1e-6,
           nb::arg("h") = 0.001)
        .def_static("run_xlbomd", [](const std::vector<int>& atomic_numbers,
                                     const std::vector<double>& init_positions,
                                     const std::vector<double>& masses,
                                     double dt, int n_steps,
                                     double grid_h, double grid_margin,
                                     int max_iter, double tol) {
            return scf::NaoDriver::RunXLBOMD(atomic_numbers, init_positions,
                                               masses, dt, n_steps,
                                               grid_h, grid_margin, max_iter, tol);
        }, nb::arg("atomic_numbers"), nb::arg("init_positions"),
           nb::arg("masses"), nb::arg("dt"), nb::arg("n_steps"),
           nb::arg("grid_h") = 0.3, nb::arg("grid_margin") = 4.0,
           nb::arg("max_iter") = 50, nb::arg("tol") = 1e-6);

    m.attr("__version__") = "0.1.0-alpha";
}
