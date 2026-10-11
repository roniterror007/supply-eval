"""
PlanFlow AI - Structured Rich Logging Subsystem
Provides colored, formatted console and file logging with agent-specific themes.
"""

import logging
import sys
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()

AGENT_COLORS = {
    "ForecastingAgent": "cyan",
    "OpportunityAgent": "magenta",
    "DemandPlanningAgent": "blue",
    "FactoryPlanningAgent": "yellow",
    "ProductionAgent": "green",
    "SupplyChainAgent": "red",
    "ScenarioAgent": "bright_cyan",
    "RecommendationAgent": "bright_magenta",
    "JudgeManagerAgent": "bold white on dark_blue",
    "ExecutiveBossAgent": "bold white on dark_red",
    "VerificationEngine": "bold yellow",
    "EvaluationEngine": "bold green",
}


def get_logger(name: str = "PlanFlowAI", level: int = logging.INFO) -> logging.Logger:
    """Returns a structured logger configured with Rich output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        formatter = logging.Formatter("[%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_agent_dialogue(sender: str, recipient: str, message: str, correction: bool = False):
    """Prints a styled friendly peer communication message to the console."""
    s_color = AGENT_COLORS.get(sender, "white")
    r_color = AGENT_COLORS.get(recipient, "white")
    tag = "[bold red][MUTUAL CORRECTION][/bold red]" if correction else "[dim cyan][PEER EXCHANGE][/dim cyan]"
    console.print(f"{tag} [{s_color}]{sender}[/{s_color}] -> [{r_color}]{recipient}[/{r_color}]:")
    console.print(f"   [italic]{message}[/italic]\n")


def log_audit_check(check_name: str, passed: bool, details: str):
    """Logs a mathematical or contract verification audit check."""
    status = "[bold green][PASS][/bold green]" if passed else "[bold red][VIOLATION][/bold red]"
    console.print(f"[bold white][JUDGE AUDIT][/bold white] {check_name}: {status} - {details}")


def log_boss_verdict(decision: str, justification: str):
    """Logs the Executive Boss final decision memo."""
    color = "bold green" if "APPROVED" in decision else ("bold yellow" if "CONDITIONAL" in decision else "bold red")
    console.rule(f"[{color}]EXECUTIVE BOSS SIGN-OFF: {decision}[/{color}]")
    console.print(f"[bold]Justification:[/bold] {justification}\n")
