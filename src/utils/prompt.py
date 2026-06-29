"""Prompt placeholder substitution helpers.

Detection, verification, and reviewer prompts all use ``{placeholder}``
substitution rather than appending content unconditionally. ``str.replace``
(not ``str.format``) is intentional so literal ``{{...}}`` JSON examples in
prompt bodies stay intact.
"""

SPONSOR_DATABASE_HEADER = (
    "\n\nDYNAMIC SPONSOR DATABASE (current known sponsors - treat as high confidence):\n"
)


def render_prompt(prompt: str, **vars: str) -> str:
    """Substitute ``{name}`` placeholders in ``prompt`` with provided values.

    Variables without a corresponding placeholder are silently dropped: that
    is the supported way for a user to opt out of an injection by removing
    the placeholder from their customized prompt.
    """
    rendered = prompt
    for name, value in vars.items():
        rendered = rendered.replace('{' + name + '}', value)
    return rendered


def format_sponsor_block(sponsor_list: str) -> str:
    """Wrap a non-empty sponsor list with the standard header.

    Empty list returns empty string so substitution does not produce a
    dangling header on prompts whose ``{sponsor_database}`` placeholder is
    left in place.
    """
    if not sponsor_list:
        return ""
    return SPONSOR_DATABASE_HEADER + sponsor_list


OVERRIDE_HEADER = "\n\nADDITIONAL INSTRUCTIONS (these take precedence):\n"


def format_override_block(override: str) -> str:
    """Wrap a non-empty per-pass override with its header.

    Empty (the default) returns an empty string, so the built-in default prompt
    renders byte-identically to today -- an override only adds content when the
    user supplies one for that pass.
    """
    if not override or not override.strip():
        return ""
    return OVERRIDE_HEADER + override


def apply_override(prompt: str, override_block: str) -> str:
    """Inject a per-pass override block into a prompt.

    If the prompt contains an ``{override}`` placeholder (a customized prompt that
    opted to control placement), substitute it there; otherwise append the block,
    which leaves the unmodified built-in defaults intact. An empty block is a no-op.
    """
    if '{override}' in prompt:
        return prompt.replace('{override}', override_block)
    return prompt + override_block if override_block else prompt


def render_with_override(rendered: str, override: str) -> str:
    """Apply an optional per-pass override to an already-rendered prompt.

    Wraps the format + apply steps so each render site only supplies the override
    text it fetched. Empty/None override is a no-op.
    """
    return apply_override(rendered, format_override_block(override or ''))
