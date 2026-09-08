"""Unit tests for the perturbation-vocabulary fix (basal reserved code 0; unassigned/control
excluded from f_pert vocab) and the two-axis perturbmap treatment."""
import numpy as np
import run_concert_spac as R
from run_concert_spac import build_attributes_spac, BASAL_PERT_KEY


def test_perturbmap_three_ko_basal_slot():
    # 3 KO genes + basal(unperturbed) + KP/periphery folded to basal (no NTC)
    perturb = np.array(["Jak2", "Tgfbr2", "Ifngr2", "unperturbed", "unperturbed"])
    tissue  = np.array(["tumor", "tumor", "tumor", "normal", "periphery"])
    _, td, pidx, pm = build_attributes_spac(
        perturb, tissue_raw=tissue,
        unassigned_labels=frozenset({"unperturbed"}), control_labels=frozenset({"__NONE__"}))
    assert pm[BASAL_PERT_KEY] == 0, pm
    assert {k for k in pm if k != BASAL_PERT_KEY} == {"Jak2", "Tgfbr2", "Ifngr2"}, pm
    # basal cells map to code 0; the 3 KO cells map to non-zero
    assert list(pidx) == [pm["Jak2"], pm["Tgfbr2"], pm["Ifngr2"], 0, 0], pidx
    assert td == {"normal": 0, "periphery": 1, "tumor": 2}, td
    print("[ok] perturbmap: 3 KO + basal slot 0, 3 tissue classes")


def test_spac_excludes_unassigned_and_control():
    # spac: real genes + unassigned(basal) + control(NTC) -> both to reserved code 0
    perturb = np.array(["GeneA", "GeneB", "unassigned", "control", "GeneA"])
    _, _, pidx, pm = build_attributes_spac(
        perturb, tissue_raw=None,
        unassigned_labels=frozenset({"unassigned"}), control_labels=frozenset({"control"}))
    assert pm[BASAL_PERT_KEY] == 0, pm
    assert {k for k in pm if k != BASAL_PERT_KEY} == {"GeneA", "GeneB"}, pm
    assert pidx[2] == 0 and pidx[3] == 0, "unassigned + control must land in basal slot 0"
    assert pidx[0] > 0 and pidx[1] > 0, "real genes must be non-zero"
    print("[ok] spac: unassigned + control excluded from vocab, -> basal slot 0")


def test_real_perts_are_sorted_1_indexed():
    perturb = np.array(["Zeta", "Alpha", "unperturbed"])
    _, _, _, pm = build_attributes_spac(
        perturb, unassigned_labels=frozenset({"unperturbed"}), control_labels=frozenset())
    assert pm == {BASAL_PERT_KEY: 0, "Alpha": 1, "Zeta": 2}, pm
    print("[ok] real perts sorted, 1-indexed after basal slot")


if __name__ == "__main__":
    test_perturbmap_three_ko_basal_slot()
    test_spac_excludes_unassigned_and_control()
    test_real_perts_are_sorted_1_indexed()
    print("ALL PASS")
