"""Dynamic skill loader for agent_computer.

Loads auto-extracted Python skill files from memory/skills/ and registers
them as tools in the ToolRegistry, making them available to the agent.
"""

from __future__ import annotations
import importlib.util
import inspect
import logging
from pathlib import Path

from tool_registry import Tool, ToolParam, ToolRegistry

logger = logging.getLogger("agent_computer.skill_loader")

# Map Python type annotations to OpenAI parameter types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def load_skills(skills_dir: Path | str, registry: ToolRegistry) -> list[str]:
    """Load all .py skill files from a directory and register as tools.

    Returns a list of successfully registered skill names.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    registered = []
    for skill_file in sorted(skills_dir.glob("*.py")):
        name = _load_single_skill(skill_file, registry)
        if name:
            registered.append(name)

    if registered:
        logger.info(f"Loaded {len(registered)} skills: {', '.join(registered)}")
    return registered


def _load_single_skill(skill_file: Path, registry: ToolRegistry) -> str | None:
    """Load a single skill file and register it as a tool.

    Returns the skill name if successful, None otherwise.
    """
    name = skill_file.stem

    # Skip if already registered
    if registry.get(name):
        logger.debug(f"Skill {name} already registered, skipping")
        return None

    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(f"skill_{name}", str(skill_file))
        if spec is None or spec.loader is None:
            logger.warning(f"Could not create module spec for {skill_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the first public async function
        handler = None
        for member_name, member in inspect.getmembers(module, inspect.isfunction):
            if member_name.startswith("_"):
                continue
            if inspect.iscoroutinefunction(member):
                handler = member
                break

        if handler is None:
            logger.warning(f"No public async function found in {skill_file}")
            return None

        # Extract description from module docstring
        doc = module.__doc__ or ""
        description = doc.strip().split("\n")[0] if doc.strip() else f"Skill: {name}"
        description = f"[skill] {description}"

        # Build params from function signature
        sig = inspect.signature(handler)
        params = []
        for param_name, param in sig.parameters.items():
            # Determine type
            annotation = param.annotation
            param_type = _TYPE_MAP.get(annotation, "string")

            # Determine if required (no default = required)
            required = param.default is inspect.Parameter.empty

            params.append(ToolParam(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
            ))

        # Register the tool
        tool = Tool(
            name=name,
            description=description,
            params=params,
            handler=handler,
        )
        registry.register(tool)
        logger.info(f"Registered skill: {name} ({len(params)} params)")
        return name

    except Exception as e:
        logger.error(f"Failed to load skill {skill_file}: {e}")
        return None
