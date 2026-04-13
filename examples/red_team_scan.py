"""Example: Run a red team scan against an LLM target."""

import asyncio

from aegis import AegisEngine
from aegis.engine import AegisConfig, TargetConfig


async def main() -> None:
    engine = AegisEngine()

    # Configure target
    config = AegisConfig(
        target=TargetConfig(
            provider="openai",
            model="gpt-4o",
            system_prompt="You are a helpful customer service assistant for Acme Corp.",
        ),
        attack_categories=[
            "prompt_injection",
            "jailbreak",
            "data_exfiltration",
        ],
        mutation_rounds=2,
        max_attacks_per_category=20,
        compliance_frameworks=["owasp_llm_top10", "nist_ai_rmf"],
    )

    # Run full assessment
    posture = await engine.assess(config)
    print(posture.summary())

    # Access detailed results
    if posture.scan_result:
        print(f"\nVulnerabilities found:")
        for result in posture.scan_result.successful:
            print(f"  [{result.attack.severity.value}] {result.attack.name}")
            print(f"    Payload: {result.payload[:80]}...")
            print(f"    Response: {result.response[:80]}...")

    if posture.compliance:
        print(posture.compliance.summary())


if __name__ == "__main__":
    asyncio.run(main())
