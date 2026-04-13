"""CLI entry point for AEGIS."""

from __future__ import annotations

import asyncio
import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="aegis-ai")
def main() -> None:
    """AEGIS — LLM Security Testing & Runtime Guardrails Engine."""
    pass


@main.command()
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic", "custom"]), default="openai")
@click.option("--model", "-m", default="gpt-4o")
@click.option("--categories", "-c", multiple=True, default=["prompt_injection", "jailbreak", "data_exfiltration"])
@click.option("--mutations", default=3, type=int, help="Mutation rounds")
@click.option("--max-attacks", default=50, type=int, help="Max attacks per category")
def scan(provider: str, model: str, categories: tuple[str, ...], mutations: int, max_attacks: int) -> None:
    """Run a red team scan against an LLM target."""
    from aegis.engine import AegisEngine, AegisConfig, TargetConfig

    config = AegisConfig(
        target=TargetConfig(provider=provider, model=model),
        attack_categories=list(categories),
        mutation_rounds=mutations,
        max_attacks_per_category=max_attacks,
    )

    async def _run() -> None:
        engine = AegisEngine()
        posture = await engine.assess(config)
        console.print(posture.summary())

    asyncio.run(_run())


@main.command()
@click.argument("text")
def classify(text: str) -> None:
    """Classify a single input for prompt injection."""
    from aegis.detection.classifier import InjectionClassifier

    classifier = InjectionClassifier()
    result = classifier.classify_detailed(text)

    label_color = {"benign": "green", "suspicious": "yellow", "malicious": "red"}.get(result.label, "white")
    console.print(f"\n[bold {label_color}]{result.label.upper()}[/bold {label_color}] "
                  f"(score: {result.score:.3f})")

    if result.signals:
        console.print("\n[bold]Signals:[/bold]")
        for signal, score in result.signals.items():
            console.print(f"  {signal}: {score:.3f}")

    if result.matched_patterns:
        console.print(f"\n[bold]Matched Patterns:[/bold]")
        for p in result.matched_patterns:
            console.print(f"  • {p}")


@main.command()
@click.argument("text")
@click.option("--threshold", "-t", default=0.7, type=float)
def guard(text: str, threshold: float) -> None:
    """Validate input through guardrails."""
    from aegis.guardrails.proxy import GuardrailsProxy

    proxy = GuardrailsProxy(block_threshold=threshold)

    async def _run() -> None:
        result = await proxy.validate_input(text)
        console.print(result.summary())

    asyncio.run(_run())


@main.command()
@click.option("--port", "-p", default=8000, type=int)
def serve(port: int) -> None:
    """Launch the AEGIS guardrails API server."""
    import uvicorn
    from aegis.api.server import create_app

    app = create_app()
    console.print(f"[bold green]Starting AEGIS API on port {port}[/bold green]")
    uvicorn.run(app, host="0.0.0.0", port=port)


@main.command()
def attacks() -> None:
    """List all attacks in the catalog."""
    from aegis.redteam.attacks import get_attack_catalog

    catalog = get_attack_catalog()
    total = 0
    for category, attack_list in catalog.items():
        console.print(f"\n[bold]{category}[/bold] ({len(attack_list)} attacks)")
        for attack in attack_list:
            console.print(f"  [{attack.severity.value}] {attack.id}: {attack.name}")
            total += 1

    console.print(f"\n[dim]Total: {total} attacks in catalog[/dim]")


if __name__ == "__main__":
    main()
