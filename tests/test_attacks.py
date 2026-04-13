"""Tests for attack catalog."""

import pytest
from aegis.redteam.attacks import (
    get_attack_catalog, get_attacks_by_category, get_attacks_by_severity,
    get_all_attacks, Attack, AttackCategory, Severity,
)


class TestAttackCatalog:
    def test_catalog_has_all_categories(self):
        catalog = get_attack_catalog()
        assert "prompt_injection" in catalog
        assert "jailbreak" in catalog
        assert "data_exfiltration" in catalog
        assert "privilege_escalation" in catalog
        assert "denial_of_service" in catalog

    def test_catalog_not_empty(self):
        catalog = get_attack_catalog()
        for category, attacks in catalog.items():
            assert len(attacks) > 0, f"{category} has no attacks"

    def test_all_attacks_have_ids(self):
        for attack in get_all_attacks():
            assert attack.id, f"Attack missing ID: {attack.name}"
            assert attack.name, f"Attack missing name: {attack.id}"
            assert attack.payload, f"Attack missing payload: {attack.id}"

    def test_all_attacks_have_category(self):
        for attack in get_all_attacks():
            assert isinstance(attack.category, AttackCategory)

    def test_all_attacks_have_severity(self):
        for attack in get_all_attacks():
            assert isinstance(attack.severity, Severity)

    def test_unique_ids(self):
        ids = [a.id for a in get_all_attacks()]
        assert len(ids) == len(set(ids)), "Duplicate attack IDs found"

    def test_get_by_category(self):
        pi = get_attacks_by_category("prompt_injection")
        assert len(pi) > 0
        assert all(a.category == AttackCategory.PROMPT_INJECTION for a in pi)

    def test_get_by_category_empty(self):
        result = get_attacks_by_category("nonexistent")
        assert result == []

    def test_get_by_severity(self):
        critical = get_attacks_by_severity(Severity.CRITICAL)
        assert len(critical) > 0
        assert all(a.severity == Severity.CRITICAL for a in critical)

    def test_total_attack_count(self):
        all_attacks = get_all_attacks()
        assert len(all_attacks) >= 20  # We defined at least 20


class TestAttackModel:
    def test_attack_fields(self):
        attack = Attack(
            id="test-001",
            name="Test Attack",
            category=AttackCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            payload="test payload",
            owasp_id="LLM01",
        )
        assert attack.id == "test-001"
        assert attack.category == AttackCategory.PROMPT_INJECTION
        assert attack.owasp_id == "LLM01"

    def test_attack_defaults(self):
        attack = Attack(
            id="t", name="t", category=AttackCategory.JAILBREAK,
            severity=Severity.LOW, payload="p",
        )
        assert attack.tags == []
        assert attack.metadata == {}
        assert attack.description == ""
